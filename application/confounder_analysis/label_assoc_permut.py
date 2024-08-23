# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from application.utils.utilIO import mkdir
from application.utils.performance_eval import bootstrap_runs_mout,auroc_folds_stats
from application.utils.statistical_tests import selectPairs
from application.project import WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT,LAYER_DICT,DATA_DIR,TISSUE_TYPES_HIST,get_experiments

EXPERIMENT = 'ConfounderAnalysis'
OUT_DIR = f'{OUTPUT_DIR}/{EXPERIMENT}/labels_assoc_cv'
LABEL_ASSOC_DIR = f'{OUTPUT_DIR}/LabelCodependence/csv'
PRED_DIR =f'{OUTPUT_DIR}/Predictions'
mkdir(OUT_DIR)

experiments = get_experiments()

#Selecting model with best AUROC
testCases = selectPairs(
                        codependence_dir=LABEL_ASSOC_DIR,
                        pred_dir=PRED_DIR,
                        experiments=experiments,
                        tissueTypes=TISSUE_TYPES_HIST
                        )
for cohort in TISSUE_TYPES_HIST:
    pairs = testCases[cohort]
    p_values = []
    auc_values = []
    props = []
    ORs = []
    aucs_cv = []
    positive_props = []
    voi_aurocs = []
    for v_ in pairs.keys():
        for lv_ in pairs[v_].keys():
            exp_tags = [pairs[v_][lv_][0]]
            cv_res = pairs[v_][lv_][1]
            for exp in exp_tags:
                pDf = pd.read_csv(f'{PRED_DIR}/tcga_{cohort}_{exp}.csv')
                VOI,LV,SV = f'T_{v_}',f'T_{lv_}',f'P_{lv_}' #names of the variable of interest, label and score variables
                # Dropping null value fields
                df = pDf.loc[:,[VOI,LV,SV]].dropna()
                voi,label,score = df[VOI], df[LV],df[SV]
                #variable of interest = voi (the variable we are intersted in analyzing e.g., the grade or CDH1 status)
                #label: true label of the predicted variable
                #score: prediction score of the predicted variable
                test_statistic = roc_auc_score#average_precision_score#
                uvals = list(set(voi))#unique values of the VOI
                N = 100000# use at least 100000
                B = np.ones((N,len(uvals)))*np.nan
                for i in range(N):
                    pg = np.random.permutation(voi)
                    for j,v in enumerate(uvals):
                        try: #to prevent errors it cannot be computed (e.g., due to having a single class)
                            B[i,j] = test_statistic(label[pg==v],score[pg==v])
                        except:
                            continue
                try:
                    # B is the background distribution obtained from the permutation test
                    F = [test_statistic(label[voi==v],score[voi==v]) for v in uvals]
                    #F contains the test statistic for the observed statistic
                    p = [2*min(np.mean(f>=B[:,i]),np.mean(f<=B[:,i])) for i,f in enumerate(F)] #comopute two sided p-val (may remove multiply with 2.0 to be less stringent)
                    p_values.append(p)
                    auc_values.append(F)
                    aucs_cv.append(cv_res)
                except:
                    print('Strong association/single label in one class')
                    p_values.append([np.nan,np.nan])
                    continue
                #%%
                print('Observed value',F)
                print('Mean/Std Background value:',np.nanmean(B,axis=0),np.nanstd(B,axis=0))
                print('p-values (for each group):',p)
                # print('p-values (for average across group):',p_avg)
                voi_aurocs.append([max(1-roc_auc_score(voi==uvals[-1],score),roc_auc_score(voi==uvals[-1],score))]) #set value here you would like to test
                positive_props.append(df.groupby(VOI)[LV].mean().to_list())
                contTable = pd.crosstab(voi,label)
                prop = contTable.to_numpy().ravel()
                props.append(prop/prop.sum())
                contTable = contTable+1e-2
                OR = np.log2((contTable.iloc[0,0]*contTable.iloc[1,1])/(contTable.iloc[0,1]*contTable.iloc[1,0]))
                ORs.append([OR])
      
    from statsmodels.stats.multitest import multipletests
    p_values = np.array(p_values)
    auc_values = np.array(auc_values)

    pval_df_indexes = np.array(
               [f'{v_}_{k}_{experiments[pairs[v_][k][0]]}'
               for v_ in pairs.keys()
               for k in pairs[v_].keys()]
               )
    _,corp,_,_ = multipletests(
                    p_values[~np.isnan(p_values).any(1)].ravel(),
                    method='fdr_bh'
                    )
    pvalDf = pd.DataFrame(np.hstack((
                                    auc_values,
                                    positive_props,
                                    corp.reshape(-1,2),
                                    voi_aurocs,
                                    props,
                                    ORs
                                    )))
    pvalDf.index = pval_df_indexes[~np.isnan(p_values).any(1)]
    pvalDf.columns = ['AUROC (WT)','AUROC (MUT)','WT +ve ratio','MUT +ve ratio','p (WT)','p (MUT)','VOI AUROC','WT-WT','WT-MUT','MUT-WT','MUT-MUT','Tendency']
    pvalDf = pvalDf.round(decimals=3)
    pvalDf.insert(loc=0,column='AUROC (STD)',value=aucs_cv)
    pvalDf.to_csv(f'{OUT_DIR}/{cohort}.csv')
