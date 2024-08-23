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

OUT_DIR = f'{OUTPUT_DIR}/Predictions'

COHORT = 'tcga'
n_folds = 4

for Repr in LAYER_DICT.keys():
    for tissue in TISSUE_TYPES_HIST:
        INFER_DIR = f'{OUTPUT_DIR}/HistPred/{COHORT}/CLAM/{Repr}/{tissue}'
        df = pd.DataFrame()
        genesDir = glob(os.path.join(INFER_DIR, f"{str.upper(tissue)}*"))
        for geneFile in genesDir:
            if '_eval' not in geneFile:continue
            fidx = []
            # Loading fold level prediction
            mutDf = pd.DataFrame()
            tissue_gene = Path(geneFile).stem
            gene = tissue_gene.split("_")[1]
            print(f'Processing {tissue_gene}')
            geneDf = pd.DataFrame()
            fidx = [] 
            for fold_idx in range(n_folds):
                # adding exception handlers ass for some mutation we don't have experimental result
                # to be fixed in the next run, as the error was cuased due to generating non-stratifed splits
                fold_res_file = f'{geneFile}/fold_{fold_idx}.csv'
                tmpDf = pd.read_csv(fold_res_file)
                geneDf = pd.concat([geneDf,tmpDf])
                fidx.extend([fold_idx]*tmpDf.shape[0]) 
            geneDf[f'{gene}_fold_idx'] = fidx
            geneDf.rename(columns={'slide_id':'Patient ID','Y':gene,'p_1':f'P_{gene}'},inplace=True)
            geneDf.drop(columns=['Y_hat','p_0'],inplace=True)
            geneDf.set_index('Patient ID',inplace=True)
            if df.shape[0]==0:
                df = geneDf
            else:
                df = df.join(geneDf)

        df.to_csv(f'{OUT_DIR}/{COHORT}_{tissue}_{Repr}_CLAM.csv')
