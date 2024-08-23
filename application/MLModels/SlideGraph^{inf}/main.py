from ast import arg
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from model.gnn import *
import argparse
from pathlib import Path
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=('Passing tissue name and'
            ,'worksapce dir and also graphs dir'))
    parser.add_argument('--gpu', type=str, nargs='?', const='0', default='0')
    parser.add_argument('--data_dir', type=str,
                         default=DATA_DIR)
    parser.add_argument('--out_dir', type=str,
                         default=f'{OUTPUT_DIR}/SlideGraph')
    parser.add_argument('--tissue', type=str,
                         default='colon')
    parser.add_argument('--wsi_meta', type=str,
                         default=f'{DATA_DIR}/wsi_meta.csv')
    parser.add_argument('--feat', type=str,
                         default='SHUFFLE')
    parser.add_argument('--dthreshold', type=int,
                         default=None)
    parser.add_argument('--graphs', type=str,
                         default=f'{OUTPUT_DIR}/GraphsDF')
    parser.add_argument('--nodeproba', type=bool,
    default=False)
    args = parser.parse_args()
    print('Printing args dictionary',args)
    # Saving Node Level Prediction or not
    returnNodeProba = args.nodeproba
    SDATA_DIR = args.data_dir
    OUTPUT_DIR = args.out_dir      
    GRAPHS_DIR = args.graphs
    FEAT = args.feat
    WSI_META = args.wsi_meta

        # Model parameters and hyperparameters
    device = f'cuda:{args.gpu}'
    cpu = 'cpu'
    lr = 0.001
    weight_decay = 0.0001
    epochs = 300#00 # Total number of epochs
    folds =4 # number of folds
    scheduler = None
    batch_size = 8
    NEIGH_RADIUS=args.dthreshold#1500
    conv = 'EdgeConv'
    TRAIN_WHOLE = False
    CV = True
    w_model_epochs = 50

    FILTER = {'Qui':False,'Uniq':False,'BAG':True,'OVERLAPED':False,'EX_MISSING_MPP':True}

    if FEAT=='SSL':
        layers = [2048,2048,2048]#,1024,1024]#,1024,1024]
    elif FEAT=='CTransPath':
        layers = [768,768,768]
    else:
        layers = [1024,1024,1024]

    # Loading GDC Manifest and Graphs
    Manifest = pd.read_csv(f'{SDATA_DIR}/META_WSIs/gdc_sample_sheet_mutation.tsv',delimiter='\t')
    Manifest.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    Manifest['Patient ID']=[Path(p).stem for p in Manifest['File Name']]
    Manifest.set_index('Patient ID',inplace=True)
    graphlist = glob(os.path.join(GRAPHS_DIR, "*.pkl"))#[0:1000]
    graphDf = pd.DataFrame(graphlist,columns=['Path'])
    graphDf['Patient ID'] = [Path(g).stem for g in graphDf['Path']]
    graphDf.set_index('Patient ID',inplace=True)

     # Excluding WSIs with missing MPP
    if FILTER['EX_MISSING_MPP']:
        metaDf = pd.read_csv(WSI_META,index_col='wsi_name')
        graphDf = graphDf.join(metaDf).dropna().loc[:,['Path']]

    mappingDf = graphDf.join(Manifest)
    projDf = mappingDf[mappingDf['Project ID']==f'TCGA-{str.upper(args.tissue)}']

    Exid = (
            f'FEAT_{FEAT}_lr_{lr}_decay_{weight_decay}_bsize_{batch_size}_layers_{"_".join(map(str,layers))}_dth_{NEIGH_RADIUS}_conv{conv}'
            f'BAG_{FILTER["BAG"]}_overlapped_{FILTER["OVERLAPED"]}_EX_MISSING_{FILTER["EX_MISSING_MPP"]}{args.tissue}'
             )
  
    MUT = pd.read_csv(f'{SDATA_DIR}/MUT/tcga_{args.tissue}.csv',index_col='patient_id')
    MUT.index = [idx.split(':')[-1][:12] for idx in MUT.index]
    # Check number of sample counts
    projMUT = projDf.set_index('Case ID').join(MUT).dropna()
    temp = projMUT[~projMUT.index.duplicated(keep='first')]
    print(temp.shape)
    projDf = projDf.set_index('Case ID').join(MUT)
    geneList = MUT.columns.tolist()

    # Training a seprate model for each mutation type
    # geneList = ['TP53']#,'IDH1']
    for voi in tqdm(geneList):
        dataset =[]
        MUTDf = projDf.loc[:,[voi,'Path']]  
        MUTDf['WSI Name'] = [Path(g).stem.split('.')[0] for g in MUTDf['Path']]
        MUTDf = MUTDf.dropna()
        for pid in tqdm(set(MUTDf.index)):
            graphs = MUTDf[MUTDf.Path.str.contains(pid)].Path.tolist()
            # if len(graphs)!=4:continue
            # print(graphs)
            # print(len(graphs))
            # Loading each graph
            gBagN = []
            gBagE = []
            gBagC = []
            for i,g in enumerate(graphs):
                G = pickleLoad(g)
                #print(G)
                    # Constructing raidus neighbour graphs
                if NEIGH_RADIUS:
                    W = radius_neighbors_graph(toNumpy(G.coords),NEIGH_RADIUS,mode='connectivity',include_self=False).toarray()
                    g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
                    g.coords = G.coords
                    G = g
                if i>0:
                    offset = torch.cat(gBagN[:i]).shape[0]
                    gBagE.append(G.edge_index+offset)
                    gBagC.append(G.coords+offset)
                else:
                    gBagE.append(G.edge_index)
                    gBagC.append(G.coords)
                gBagN.append(G.x)
            G.edge_index = torch.cat(gBagE,1)
            G.x = torch.cat(gBagN)
            G.coords = torch.cat(gBagC)
            G.pid = pid

            # In case of bag we will have multiple entries
            if len(graphs)>1:
                tStatus = float(MUTDf.loc[pid,voi].tolist()[i])
                # import pdb; pdb.set_trace()
            else:
                tStatus = float(MUTDf.loc[pid,voi])
            G.y = toTensor([tStatus], dtype=torch.float32, requires_grad=False)
            dataset.append(G)
            TEMP=False
            if TEMP:
                dict_coords = {'c'+str(i):G.coords[:,i] for i in range(G.coords.shape[1])}
                dict_feats = {'f'+str(i):G.x[:,i] for i in range(5)}
                dict_y = {'y'+str(i):G.x[:,i] for i in range(5)}
                node_dict = {**dict_coords, **dict_feats,**dict_y}
                d = Data(**node_dict,edge_index = G.edge_index, edge_attr = G.edge_attr)
                nG = to_networkx(d, node_attrs=list(node_dict.keys()))
                nx.write_gml(nG,f'{OUTPUT_DIR}/gml_test_rad_neigh_2000_4.gml')
 
        TS = MUTDf.loc[:,[voi]]
        print(len(dataset),TS[~TS.index.duplicated(keep='first')].shape)

        RR = np.full_like(np.zeros((folds, TS.shape[1], 2)),np.nan)
        
        SS = pd.DataFrame([[float(G.y),G.pid] for G in dataset],columns=[voi,'pid'])
        SS = SS[~SS.pid.duplicated(keep='first')]

        print('GENE: ',voi,
            "# of MUT cases",sum(SS[voi]==1),' Wild Type ',sum(SS[voi]==0))
        
        # Stratified cross validation
        skf = StratifiedKFold(n_splits=folds, shuffle=True)

        Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [
        ], [], [], [], [],  []  # Intialise outputs

        fold = 0
        for trvi, test in skf.split(SS.loc[:,'pid'],SS.loc[:,voi]):
            train, valid = train_test_split(
                trvi, test_size=0.10, shuffle=True,stratify=SS.iloc[trvi,0])  # ,
            
            # selecting Training samples
            train_patients =SS.iloc[train,1].tolist()
            valid_patients = SS.iloc[valid,1].tolist()
            test_patients = SS.iloc[test,1].tolist()

            # Check for data leakage
            if len(set(train_patients).intersection(set(valid_patients)).intersection(set(test_patients)))>0:
                print('Data Mixing between train test and validation splits')
                exit()

            train_dataset = [dataset[i] for i in range(len(dataset)) if dataset[i].pid in train_patients]
            valid_dataset = [dataset[i] for i in range(len(dataset)) if dataset[i].pid in valid_patients]
            test_dataset = [dataset[i] for i in range(len(dataset)) if dataset[i].pid in test_patients]

            if conv=='PNAConv':
                # Compute the maximum in-degree in the training data.
                deg = compute_degree(train_dataset)
            else:
                deg=0

            v_loader = DataLoader(valid_dataset, shuffle=False)
            tt_loader = DataLoader(test_dataset, shuffle=False)

            model = GNN(dim_features=dataset[0].x.shape[1], dim_target=TS.shape[1],
                        degree=deg,
                        layers=layers, dropout=0.1, pooling='mean', conv=conv, aggr='max',
                        device=device)

            net = NetWrapper(model, loss_function=None,
                            device=device,batch_size=batch_size)
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay)
            
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            Q, train_loss, train_acc, val_acc, tt_acc, val_pr, test_pr = net.train(
                train_loader=train_dataset,
                max_epochs=epochs,
                optimizer=optimizer,
                scheduler=None,#ReduceLROnPlateau(optimizer, 'min'),
                clipping=None,
                validation_loader=v_loader,
                test_loader=tt_loader,
                early_stopping=20,
                return_best=False,
                log_every=5)
            # Fdata.append((best_model, test_dataset, valid_dataset))
            Vacc.append(val_acc)
            Tacc.append(tt_acc)
            Vapr.append(val_pr)
            Tapr.append(test_pr)
            print("\nfold complete", len(Vacc), train_acc,
                val_acc, tt_acc, val_pr, test_pr)

            print('.....'*20,'Saving Convergence Curve','........'*20)

            path_plot_conv = f'{OUTPUT_DIR}/Converg_Curves/{Exid}/{voi}'
            mkdirs(path_plot_conv)
            import matplotlib.pyplot as plt
            ep_loss = np.array(net.history)
            plt.plot(ep_loss)#[:,0]); plt.plot(ep_loss[:,1]); plt.legend(['train','val']);
            plt.savefig(f'{path_plot_conv}/{len(Vacc)}.png')
            plt.close()

            print('.....'*20,'Saving Best model Weights','........'*20)
            weights_path = f'{OUTPUT_DIR}/Weights/{Exid}/{voi}'
            mkdirs(weights_path)

            torch.save(Q[0][0].state_dict(), f'{weights_path}/{fold}')

            # Saving node level predictions
            zz, yy, zxn, pn = EnsembleDecisionScoring(
                Q, test_dataset, device=net.device, k=10) # Using 10 ensemble models.

            # saving ensemble results for each fold
            n_classes = zz.shape[-1]
            R = np.full_like(np.zeros((n_classes,2)),np.nan)

            for i in range(n_classes):
                try:
                    R[i] = np.array(
                        [calc_roc_auc(yy[:, i], zz[:, i]), calc_pr(yy[:, i], zz[:, i])])
                except:
                    print('only one class') 

            df = pd.DataFrame(R, columns=['AUROC', 'AUC-PR'])
            df.index = TS.columns
            RR[fold] = R

            res_dir = f'{OUTPUT_DIR}/Results/{Exid}/{voi}'
            mkdirs(res_dir)
            df.to_csv(f'{res_dir}/{fold}.csv')
            print(df)

            node_pred_dir = f'{OUTPUT_DIR}/nodePredictions/{Exid}/{voi}'
            mkdirs(node_pred_dir)

            # saving results of fold prediction
            foldPred = np.hstack((pn[:, np.newaxis], zz, yy))
            foldPredDir = f'{OUTPUT_DIR}/foldPred/{Exid}/{voi}'
            mkdirs(foldPredDir)

            columns = ['Patient ID'] +[f'P_{voi}' for voi in TS.columns] + [f'T_{voi}' for voi in TS.columns]

            foldDf = pd.DataFrame(foldPred, columns=columns)
            foldDf.set_index('Patient ID', inplace=True)
            foldDf.to_csv(f'{foldPredDir}/{fold}.csv')

            if returnNodeProba:
                for i, GG in enumerate(tqdm(test_dataset)):
                    G = Data(edge_index = GG.edge_index,y=GG.y,pid=GG.pid)
                    G.to(cpu)
                    G.nodeproba = zxn[i][0]
                    # adding the target name
                    G.class_label = voi
                    ofile = f'{node_pred_dir}/{G.pid}.pkl'
                    with open(ofile, 'wb') as f:
                        pickle.dump(G, f)   
            
            # incrementing the fold number
            fold+=1
        # Averaged results of 5 without ensembling
        print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
        print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
        print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
        print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))
        # import pdb; pdb.set_trace()
        import gc; gc.collect()
        RRm = np.nanmean(RR,0)
        RRstd = np.nanstd(RR,0)
        results = pd.DataFrame(np.hstack((RRm, RRstd)))
        results.columns = ['AUROC-mean', 'AUC-PR-mean', 'AUROC-std', 'AUC-PR-std']
        results.index = TS.columns.tolist()
        results.to_csv(f'{OUTPUT_DIR}/Results/{Exid}/{voi}/{folds}_cv.csv')
        print('Results written to csv on disk')
        print(results)