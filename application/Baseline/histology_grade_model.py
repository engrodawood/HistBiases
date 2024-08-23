import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from pathlib import Path
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR,LAYER_DICT,TISSUE_TYPES_HIST
from application.utils.utilIO import mkdir
from application.utils.data_utils import load_histology
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.svm import LinearSVC


from sklearn.preprocessing import OneHotEncoder

TRAIN_COHORT = 'tcga'
Repr = 'SHUFFLE'
OUT_DIR = f'{OUTPUT_DIR}/Predictions'
mkdir(OUT_DIR)
n_folds = 4
from copy import deepcopy

def load(tissue='colon',
         cohort=TRAIN_COHORT):
    in_file = f'{OUT_DIR}/{cohort}_{tissue}_{Repr}_SGraph.csv'
    out_file = f'{OUT_DIR}/{cohort}_{tissue}_Grade.csv'
    try:
        df = pd.read_csv(in_file,index_col='Patient ID')
    except:
        df = pd.read_csv(in_file,index_col='Identifier')
    hDf = load_histology(cohort=cohort,
                         tissue=tissue)
    cDf = hDf.join(df)
    targets = list({c for c in cDf.columns if '_' not in c and c not in ['Grade']})

    return deepcopy(df),cDf,targets,out_file

def one_hot_feats(df,x=['Grade'],y=[]):
    X,Y = OneHotEncoder().fit_transform(df[x].to_numpy()),df[y].to_numpy().astype('int')
    return X,(Y*2-1).ravel() 

INFER_EXTR = 'abctb'#'cptac'#False#'abctb'

for tissue in ['brca']:

    dfc,cDf,targets,out_file = load(tissue=tissue,
                                    cohort=TRAIN_COHORT
                                    )
    if INFER_EXTR:
        dfce,cDfe,targetse,out_filee = load(tissue=tissue,
                                    cohort=INFER_EXTR
                                    )
        aucse = pd.DataFrame(np.zeros((len(targetse),n_folds)),index=targetse,columns=np.arange(0,n_folds))
    
    aucs = pd.DataFrame(np.zeros((len(targets),n_folds)),index=targets,columns=np.arange(0,n_folds))
    

    for t in targets:
        for fold in range(n_folds):
            reg = LinearSVC()
            trDf = cDf.loc[cDf[f'{t}_fold_idx']!=fold,[t,'Grade']].dropna()
            Xtr,Ytr = one_hot_feats(trDf,y=[t])
            tsDf = cDf.loc[cDf[f'{t}_fold_idx']==fold,[t,'Grade']].dropna()
            Xts,Yts = one_hot_feats(tsDf,y=[t])
            # Converting to + and -
            reg.fit(Xtr,Ytr)
            pred = reg.decision_function(Xts)
            # Updating the recods based on grade prediction
            dfc.loc[tsDf.index,[f'P_{t}']] = pred
            aucs.loc[t,fold] = roc_auc_score(Yts,pred)
            if INFER_EXTR:
                if t in targetse:
                    teDf = cDfe.loc[:,[t,'Grade']].dropna()
                    Xe,Ye = one_hot_feats(teDf,y=[t])
                    pe = reg.decision_function(Xe)
                    dfce.loc[teDf.index,[f'P_{t}']] = pe
                    aucse.loc[t,fold] = roc_auc_score(Ye,pe)  
                    print(aucse.mean(1))    
    print(aucs.mean(1))
    dfc.to_csv(out_file)
    if INFER_EXTR:
        dfce.to_csv(out_filee)

