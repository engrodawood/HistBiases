import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from pathlib import Path
import sys
sys.path.append('.')
import scipy.stats as st
print()
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR,LAYER_DICT,TISSUE_TYPES_HIST
from application.utils.utilIO import mkdir,pickleLoad

OUT_DIR = '/data/PanCancer/HistBiases/output/Predictions'

COHORT = 'abctb'
n_folds = 4

for Repr in LAYER_DICT.keys():
    for tissue in TISSUE_TYPES_HIST:
        INFER_DIR = f'{OUTPUT_DIR}/HistPred/{COHORT}/CLAM/{Repr}/{tissue}'
        df = pd.DataFrame()
        genesDir = glob(os.path.join(INFER_DIR, f"{str.upper(tissue)}*"))
        for geneFile in genesDir:
            if '_eval' not in geneFile:continue
            fidx = []
            tissue_gene = Path(geneFile).stem
            gene = tissue_gene.split("_")[1]
            print(f'Processing {tissue_gene}')
            geneDf = pd.DataFrame()
            tmpDf = pd.DataFrame()
            fidx = [] 
            for fold_idx in range(n_folds):
                # adding exception handlers ass for some mutation we don't have experimental result
                # to be fixed in the next run, as the error was cuased due to generating non-stratifed splits
                fold_res_file = f'{geneFile}/fold_{fold_idx}.csv'
                tmpDf = pd.read_csv(fold_res_file,index_col='slide_id')
                if geneDf.shape[0]!=0:
                    geneDf.loc[tmpDf.index,fold_idx] = tmpDf.loc[tmpDf.index,'p_1']
                else:
                    geneDf = tmpDf.loc[:,['p_1']]
                    geneDf.rename(columns={'p_1':fold_idx},inplace=True)
            geneDf[f'P_{gene}'] = geneDf.max(1).tolist()
            geneDf.loc[tmpDf.index,f'{gene}'] = tmpDf.loc[tmpDf.index,'Y']
            geneDf.drop(columns = np.arange(0,n_folds),inplace=True)
            geneDf.index.rename('Patient ID',inplace=True)
            if df.shape[0]==0:
                df = geneDf
            else:
                df = df.join(geneDf)
        from sklearn.metrics import roc_auc_score
        
        print('Representation',Repr)
        fields = list({c.split('_')[-1] for c in df.columns.tolist()})
        for f in fields:
            try:
                auc = roc_auc_score(df[f'{f}'].dropna(),df[f'P_{f}'].dropna())
                print(f,auc)
            except:
                print('Failed ',f)
        df.to_csv(f'{OUT_DIR}/{COHORT}_{tissue}_{Repr}_CLAM.csv')
