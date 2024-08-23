from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU, Tanh, LeakyReLU, ELU, SELU, GELU,Sigmoid
from torch_geometric.nn import GINConv, EdgeConv,PNAConv, DynamicEdgeConv, global_add_pool, global_mean_pool, global_max_pool
from tqdm import tqdm
import torch.nn as nn
import os
# Changing to parent directory for acessing the application folder
import sys
sys.path.append('.')
#sys.path.append('cloud_workspace/PanCancerAnalysis/MutPrediction')
print(sys.path)
print(os.listdir(sys.path[-1]))
from utils import (toGeometricWW,toNumpy,toTensor,DataLoader,np,torch,
                                     StratifiedShuffleSplit,time,calc_pr,calc_roc_auc,deepcopy,device,
                                     pickleLoad,pickle,mkdirs,Data,nx,to_networkx,compute_degree)
# %% Graph Neural Network


class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[16, 16, 8],degree=None,pooling='max', dropout=0.0, conv='GINConv',device='cuda:0',
                  gembed=False, **kwargs):
        """

        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.device=device
        self.nns = []
        self.convs = []
        self.linears = []
        self.featd = 32
        self.degree = degree
        self.pooling = {'max': global_max_pool,
                        'mean': global_mean_pool, 'add': global_add_pool}[pooling]
        # if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores
        self.gembed = gembed

        self.fc = Sequential(
             Linear(2, dim_target), #Sigmoid(),    
         )

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim),GELU())
                self.linears.append(
                    Sequential(
                        #Updated latest
                        #Linear(out_emb_dim, self.featd), BatchNorm1d(self.featd),ELU(),
                        Linear(dim_features, dim_target), BatchNorm1d(dim_target),ELU(),
                    )
                )

            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(
                    Sequential(
                        Linear(out_emb_dim, dim_target), BatchNorm1d(dim_target),ELU(),
                        # Linear(self.featd, dim_target), BatchNorm1d(dim_target),
                        # Linear(out_emb_dim, self.featd), BatchNorm1d(self.featd),ELU(),
                        # Linear(self.featd, dim_target), BatchNorm1d(dim_target),
                    )
                )
                if conv == 'GINConv':
                    subnet = Sequential(
                        Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ELU(),
                        #Linear(self.featd, out_emb_dim), BatchNorm1d(out_emb_dim)#, ELU(),
                        
                        )
                    self.nns.append(subnet)
                    # Eq. 4.2 eps=100, train_eps=False
                    # import pdb; pdb.set_trace()
                    self.convs.append(GINConv(self.nns[-1], **kwargs))
                elif conv == 'EdgeConv':
                    subnet = Sequential(
                        Linear(2*input_emb_dim,out_emb_dim), BatchNorm1d(out_emb_dim), ELU(),
                        # Linear(self.featd,out_emb_dim),BatchNorm1d(out_emb_dim)#,ELU(),
                        )
                    self.nns.append(subnet)
                    # DynamicEdgeConv#EdgeConv                aggr='mean'
                    self.convs.append(EdgeConv(self.nns[-1], **kwargs))
                
                elif conv == 'PNAConv':
                    aggregators = ['mean', 'min', 'max', 'std']
                    scalers = ['identity', 'amplification', 'attenuation']
                    pnaConv = PNAConv(in_channels=input_emb_dim, out_channels=out_emb_dim,
                           aggregators=aggregators, scalers=scalers, deg=self.degree,
                           edge_dim=0, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
                    subnet = Sequential(
                        Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ELU(),
                        #Linear(self.featd, out_emb_dim), BatchNorm1d(out_emb_dim)#, ELU(),
                        
                        )
                    self.nns.append(subnet)
                    self.convs.append(pnaConv)

                else:
                    raise NotImplementedError

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # has got one more for initial input
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        out = 0
        pooling = self.pooling
        Z = 0
        import torch.nn.functional as F
        for layer in range(self.no_layers):
            if layer == 0:
                #x = self.first_h(x)
                # import pdb; pdb.set_trace()
                z = self.linears[layer](x)
                Z += z
                dout = F.dropout(pooling(z, batch),
                                 p=self.dropout, training=self.training)
                out += dout
                # 
            else:
                x = self.convs[layer-1](x, edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z += torch.multiply(torch.functional.F.softmax(z,0),z)
                    if 0:
                        statFeat = torch.zeros((data.num_graphs,2))
                        for bidx in range(dout.shape[0]):
                            statFeat[bidx,:] = torch.tensor([
                                            #x[batch==bidx].max(),
                                            #x[batch==bidx].min(),
                                            x[batch==bidx].mean(),
                                            x[batch==bidx].std()
                                            ])
                        # import pdb; pdb.set_trace()
                        dout = F.dropout(
                            self.fc(statFeat.to(self.device))
                        )
                    else:
                        dout = F.dropout(pooling(z, batch),
                                        p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](
                        pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout
        
        # out = self.fc(out)

        return out, Z, x

def decision_function(model, loader, device='cpu', outOnly=False, returnNumpy=True):
    """
    generate prediction score for a given model
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    loader : TYPE Dataset or dataloader
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    outOnly : TYPE, optional 
        DESCRIPTION. The default is True. Only return the prediction scores.
    returnNumpy : TYPE, optional
        DESCRIPTION. The default is False. Return numpy array or ttensor
    Returns
    -------
    Z : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    ZXn : TYPE
        DESCRIPTION. Empty unless outOnly is False
    """
    if type(loader) is not DataLoader:  # if data is given
        loader = DataLoader(loader)
    # if type(device) == type(''):
    #     device = torch.device(device)
    ZXn = []
    model.eval()
    Pn = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            model.to(device)
            output, zn, xn = model(data)
            Pn.extend(data.pid)
            if returnNumpy:
                zn, xn = toNumpy(zn), toNumpy(xn)
            if not outOnly:
                ZXn.append((zn, xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z, Y, Pn = toNumpy(Z), toNumpy(Y), toNumpy(Pn)

    return Z, Y, ZXn, Pn


def EnsembleDecisionScoring(Q, test_dataset, device='cpu', k=None):
    """
    Generate prediction scores from an ensemble of models 
    First scales all prediction scores to the same range and then bags them
    Parameters
    ----------
    Q : TYPE reverse deque or list or tuple
        DESCRIPTION.  containing models or output of train function
    train_dataset : TYPE dataset or dataloader 
        DESCRIPTION.
    test_dataset : TYPE dataset or dataloader 
        DESCRIPTION. shuffle must be false
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    k : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    Z : Numpy array
        DESCRIPTION. Scores
    yy : Numpy array
        DESCRIPTION. Labels
    """

    Z = 0
    if k is None:
        k = len(Q)
    for i, mdl in enumerate(Q):
        if type(mdl) in [tuple, list]:
            mdl = mdl[0]
        zz, yy, ZXn, Pn = decision_function(mdl, test_dataset, device=device)
        Z += zz
        if i+1 == k:
            break
    Z = Z/k
    try:
        tem = yy.shape[1]
    except:
        yy = yy[:,np.newaxis]
    return toNumpy(Z), yy,ZXn, toNumpy(Pn)
# %%

class NetWrapper:
    def __init__(self, model, loss_function=nn.BCEWithLogitsLoss(), device='cuda:1', classification=True,batch_size=32):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
        self.batch_size = batch_size
        self.n_classes = 1 # By default the number of classes are 1

    def _pair_train(self, train_loader, optimizer, clipping=None):
        """
        Performs pairwise comparisons with ranking loss
        """

        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all = 0
        assert self.classification
        pair_count = []

        YY = toNumpy(torch.cat([d.y for d in train_loader], dim=0).float())
        
        #Checking how many classes we have
        try: 
            self.n_classes = YY.shape[1]
        except:
            YY = YY[:,np.newaxis]

        for _ in range(int(YY.shape[0]/self.batch_size)+1):
            tt = np.random.randint(self.n_classes)  # pick random task
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.batch_size)
            # handle nans in YY[:,tt] so that nans are not sampled
            vidx = np.nonzero(~np.isnan(YY[:, tt]))[0]
            bidx = vidx[list(sss.split(YY[vidx], YY[vidx, tt]))[0][1]]
            # bidx = [idx for _,idx in sss.split(YY,YY[:,tt])][0] #stratify withrespect to a random task
            data = [data for data in DataLoader(
                [train_loader[i] for i in bidx], batch_size=len(bidx))][0]
            data = data.to(self.device)
            target = data.y
            if self.n_classes==1:
                target = target.unsqueeze(1)
            if not len(pair_count):
                pair_count = np.ones(self.n_classes)
            optimizer.zero_grad()
            ypred, _, _ = model(data)
            if self.loss_fun:
                loss = self.loss_fun(ypred, target.float())
            else:
                loss_pairs = torch.zeros(self.n_classes)
                for tid in range(self.n_classes):
                    y = target[:, tid]
                    output = ypred[:, tid]

                    # take only examples with valid labels
                    vidx = ~torch.isnan(y)
                    if len(y)-len(vidx):
                        import pdb
                        pdb.set_trace()
                    output = output[vidx]
                    y = y[vidx]
                    z = toTensor([0])
                    dY = (y.unsqueeze(1)-y)
                    dZ = (output.unsqueeze(1)-output)[dY > 0]
                    dY = dY[dY > 0]
                    # import pdb;pdb.set_trace()
                    if len(dY):
                        zz = torch.zeros((2,len(dY)))
                        zz[1] = dY-dZ
                        closs = torch.mean(torch.logsumexp(10*zz,0)/10)
                        #closs = torch.mean(torch.max(z, dY-dZ))
                    else:
                        closs = toTensor(0)
                    pair_count[tid] += len(dY)
                    loss_pairs[tid] = closs
                loss = torch.mean(loss_pairs)
            acc = loss
            loss.backward()
            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            acc_all += acc.item() * num_graphs

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
        # print(pair_count)
        #import pdb;pdb.set_trace()
        return acc_all, loss_all

    def classify_graphs(self, loader):
        Z, Y, _, _ = decision_function(self.model, loader, device=self.device)

        R = np.full_like(np.zeros((self.n_classes, 2)),np.nan)

        if self.n_classes==1:
            Y = Y[:,np.newaxis]
        for i in range(self.n_classes):
            try:
                vidx = ~np.isnan(Y[:,i])
                R[i] = np.array(
                    [calc_roc_auc(Y[vidx, i], Z[vidx, i]), calc_pr(Y[vidx, i], Z[vidx, i])])
            except:
                print('Only one class')

        return np.nanmedian(R,0)[0], np.nanmedian(R,0)[1]
    
    def train_whole(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, 
                    clipping=None):
        """

        Parameters
        ----------
        train_loader : TYPE
            Training data loader.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.Adam.
        log_every : TYPE, optional
            DESCRIPTION. The default is 0.
        Returns
        -------
        trained model
        """
        
        time_per_epoch = []
        self.history = []
        iterator = tqdm(range(1, max_epochs+1))
        for epoch in iterator:
            start = time.time()

            _, train_loss = self._pair_train(
                train_loader, optimizer, clipping)

            end = time.time() - start
            time_per_epoch.append(end)
            self.history.append(train_loss)
            
        return self.model
    
    
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=100, return_best=True, log_every=0):
        """

        Parameters
        ----------
        train_loader : TYPE
            Training data loader.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.Adam.
        scheduler : TYPE, optional
            DESCRIPTION. The default is None.
        clipping : TYPE, optional
            DESCRIPTION. The default is None.
        validation_loader : TYPE, optional
            DESCRIPTION. The default is None.
        test_loader : TYPE, optional
            DESCRIPTION. The default is None.
        early_stopping : TYPE, optional
            Patience  parameter. The default is 100.
        return_best : TYPE, optional
            Return the models that give best validation performance. The default is True.
        log_every : TYPE, optional
            DESCRIPTION. The default is 0.
        Returns
        -------
        Q : TYPE: (reversed) deque of tuples (model,val_acc,test_acc)
            DESCRIPTION. contains the last k models together with val and test acc
        train_loss : TYPE
            DESCRIPTION.
        train_acc : TYPE
            DESCRIPTION.
        val_loss : TYPE
            DESCRIPTION.
        val_acc : TYPE
            DESCRIPTION.
        test_loss : TYPE
            DESCRIPTION.
        test_acc : TYPEimport pdb; pdb.set_trace()
        """

        from collections import deque
        Q = deque(maxlen=10)  # queue the last 10 models
        return_best = return_best and validation_loader is not None
        val_loss, val_acc,val_pr = -1, -1,-1
        best_val_acc, test_acc_at_best_val_acc, val_pr_at_best_val_acc, test_pr_at_best_val_acc = -1, -1, -1, -1
        test_loss, test_acc,test_pr = None, None,-1
        time_per_epoch = []
        self.history = []
        patience = early_stopping
        best_epoch = np.inf
        iterator = tqdm(range(1, max_epochs+1))
        for epoch in iterator:
            updated = False

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()

            train_acc, train_loss = self._pair_train(
                train_loader, optimizer, clipping)

            end = time.time() - start
            time_per_epoch.append(end)
            if validation_loader is not None:
                val_acc, val_pr = self.classify_graphs(
                    validation_loader)
            if test_loader is not None:
                test_acc, test_pr = self.classify_graphs(
                    test_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val_acc = test_acc
                val_pr_at_best_val_acc = val_pr
                test_pr_at_best_val_acc = test_pr
                best_epoch = epoch
                updated = True
                if return_best:
                    # import pdb; pdb.set_trace()
                    best_model = deepcopy(self.model)
                    Q.append((best_model, best_val_acc, test_acc_at_best_val_acc,
                             val_pr_at_best_val_acc, test_pr_at_best_val_acc))

                if False:
                    from vis import showGraphDataset, getVisData
                    fig = showGraphDataset(getVisData(
                        validation_loader, best_model, self.device, showNodeScore=False))
                    plt.savefig(f'./figout/{epoch}.jpg')
                    plt.close()

            if not return_best:
                # import pdb; pdb.set_trace()
                Q.append((deepcopy(self.model), val_acc,
                         test_acc, val_pr, test_pr))

            showresults = False
            if log_every == 0:  # show only if validation results improve
                showresults = updated
            elif (epoch-1) % log_every == 0:
                showresults = True

            if showresults:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR perf: {train_acc}, VL perf: {val_acc} ' \
                    f'TE perf: {test_acc}, Best: VL perf: {best_val_acc} TE perf: {test_acc_at_best_val_acc} VL pr: {val_pr_at_best_val_acc} TE pr: {test_pr_at_best_val_acc}'
                tqdm.write('\n'+msg)
            self.history.append(train_loss)

            if epoch-best_epoch > patience:
                iterator.close()
                break

        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc
            val_pr = val_pr_at_best_val_acc
            test_pr = test_pr_at_best_val_acc

        Q.reverse()
        return Q, train_loss, train_acc, np.round(val_acc, 2), np.round(test_acc, 2), val_pr, test_pr