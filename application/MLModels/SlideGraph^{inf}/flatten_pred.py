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


COHORT = 'tcga'
FOLD_PRED_DIR = f'{OUTPUT_DIR}/HistPred/{COHORT}/SlideGraph/foldPred'

for Repr in LAYER_DICT.keys():
    TAG = f'FEAT_{Repr}_lr_0.001_decay_0.0001_bsize_8_layers_{LAYER_DICT[Repr]}_dth_None_convEdgeConvBAG_True_overlapped_False_EX_MISSING_True'
    n_folds = 4

    OUT_DIR = f'{OUTPUT_DIR}/Predictions'
    mkdir(OUT_DIR)

    n_folds = 4
    for tissue in TISSUE_TYPES_HIST:
        geneList = os.listdir(f'{FOLD_PRED_DIR}/{TAG}{tissue}')
        df = pd.DataFrame()
        for gene in sorted(geneList):
            fidx = []
            geneDf = pd.DataFrame()
            for fold_idx in range(n_folds):
                tmpDf = pd.read_csv(f'{FOLD_PRED_DIR}/{TAG}{tissue}/{gene}/{fold_idx}.csv',index_col = 'Patient ID')
                geneDf = pd.concat([geneDf,tmpDf])
                fidx.extend([fold_idx]*tmpDf.shape[0]) 
            geneDf[f'{gene}_fold_idx'] = fidx  
            if df.shape[0]==0:
                df = geneDf
            else:
                df = df.join(geneDf)
        # Renaming the columns and removing " status"
        df.columns = [c.replace(' status','') for c in df.columns]
        df.columns = [c.split('_')[1].strip() if 'T_' in c else c for c in df.columns]
        df.to_csv(f'{OUT_DIR}/{COHORT}_{tissue}_{Repr}_SGraph.csv')
