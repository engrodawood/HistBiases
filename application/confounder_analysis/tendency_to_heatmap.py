import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from application.project import CODEP_COLORS,TISSUE_TYPES_HIST,CODEP_PLOT_DICT,OUTPUT_DIR,get_experiments
from application.misc.meta import COHORT_GENES_AUC,COHORT_GENES_TMB
from application.misc.colors import PERCEPTIVE_COLORS
from application.utils.plot_utils import gen_pie,mkdir
from application.utils.performance_eval import auroc_folds_stats,bootstrap_runs_mout
import math


PRED_DIR =f'{OUTPUT_DIR}/Predictions'
CONFOUND_DIR = f'{OUTPUT_DIR}/ConfounderAnalysis/TMB_HL'

OUT_DIR = f'{CONFOUND_DIR}/plots'
mkdir(OUT_DIR)

cohort = 'cptac'
hDf = pd.DataFrame()
for tissue in ['ucec']:#TISSUE_TYPES_HIST:
    tDf = pd.DataFrame()
    for cohort in ['tcga','cptac']:
        df = pd.read_csv(f'{CONFOUND_DIR}/{cohort}_{tissue}_nan_fixed.csv')
        #Grouping values based on varaile of interest
        df['PRED'] = [idx.split('_')[0] for idx in df['Unnamed: 0']]
        df.set_index('PRED',inplace=True)
        #df = df.loc[COHORT_GENES_TMB[f'{cohort}_{tissue}'],:]
        #selected = ['ER','PR','HER2']
        #selected = ['PTEN','TP53','CTCF','ARID1A','KRAS']
        selected = ['BRAF', 'TP53', 'APC', 'KRAS','PIK3CA']
        selected = ['PTEN', 'TP53', 'CTNNB1', 'ARID1A','RNF43']
        df = df.loc[selected,:]
        # Reseting the index
        df = df.reset_index()
        tDf[f'{cohort}'] = df.loc[:,'Tendency'].to_numpy()
    tDf['Tissue'] = tissue
    #tDf['Genes'] = COHORT_GENES_TMB[f'{cohort}_{tissue}']
    if hDf.shape[0]==0:
        hDf = tDf
    else:
        hDf = pd.concat([hDf,tDf])

TMP=True # Loading Codepdnece in DFCI



import matplotlib.pyplot as plt
fsize = CODEP_PLOT_DICT['FONT_SIZE']
hDf['Genes'] = selected
hDf.set_index('Genes',inplace=True)
import seaborn as sns
mapping = dict(zip(TISSUE_TYPES_HIST,np.arange(1,len(TISSUE_TYPES_HIST)+1)))
hDf['Tissue'] = [mapping[t] for t in hDf['Tissue']]
lut = dict(zip(mapping.values(),PERCEPTIVE_COLORS))
col_colors = hDf['Tissue'].map(lut)
col_colors.index = hDf.index.tolist()
g = sns.clustermap(hDf.iloc[:,[0,1]].T,cmap='seismic',row_cluster=False,method='ward',col_cluster=False,col_colors=col_colors,colors_ratio=0.12,
                    cbar_kws={"shrink": 0.5},figsize=(8,2),vmax=6,vmin=-6)
# g.ax_heatmap.set_yticklabels([])
# g.ax_heatmap.yaxis.set_ticks_position('none') 
# from matplotlib.patches import Patch

# handles = [Patch(facecolor=lut[name]) for name in lut]
# plt.legend(handles, mapping.keys(), title='Tissue',
#            bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/{cohort}_{tissue}_tend_leg_big.png',dpi=600,bbox_inches='tight')

