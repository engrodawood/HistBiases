import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def auroc_folds_stats(dataDf,targets,
                      folds = 4):
    aurocDf = pd.DataFrame(
                    np.zeros((folds,len(targets)))*np.nan,
                    columns=targets,
                    index=np.arange(0,folds)
    )
    
    for k in targets:
        tmpDf = dataDf.loc[:,[f'P_{k}',k,f'{k}_fold_idx']].dropna()
        for f_idx in range(folds):
            try:
                
                aurocDf.loc[f_idx,k] = roc_auc_score(
                                    tmpDf.loc[tmpDf[f'{k}_fold_idx']==f_idx,k].tolist(),
                                    tmpDf.loc[tmpDf[f'{k}_fold_idx']==f_idx,f'P_{k}'].tolist()
                                    )
            except:
                print('single label')
    aurocDf.columns = [t.split('(')[0] for t in aurocDf.columns]
    return aurocDf
    

def bootstrap_runs_mout(dataDf, targets,runs=1000):
    import tqdm
    rng = np.random.RandomState()
    aurocDf = pd.DataFrame(
                    np.zeros((runs,len(targets)))*np.nan,
                    columns=targets,
                    index=np.arange(0,runs)
    )
    
    for r in tqdm.tqdm(range(runs)):
        #Computing Featurewise AUROC
        for k in targets:
            tmpDf = dataDf.loc[:,[f'P_{k}',k]].dropna()
            train_patients = list(rng.choice(tmpDf[tmpDf[k]==0].index, size = sum(tmpDf[k]==0),replace=True)) +\
                list(rng.choice(tmpDf[tmpDf[k]==1].index, size = sum(tmpDf[k]==1),replace=True))
            test_patients = list(set(tmpDf.index.tolist())-set(train_patients))
            try:
                aurocDf.loc[r,k] = roc_auc_score(
                                    tmpDf.loc[test_patients,k].tolist(),
                                    tmpDf.loc[test_patients,f'P_{k}'].tolist()
                                    )
                #aurocValDf.loc[r,k] = roc_auc_score(Yv[:,idx],Yvpred[:,idx])
            except:
                print('single label')
    aurocDf.columns = [t.split('(')[0] for t in aurocDf.columns]
    return aurocDf