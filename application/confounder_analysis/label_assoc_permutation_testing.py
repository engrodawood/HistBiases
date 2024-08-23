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
from application.utils.statistical_tests import selectPairs,selectPairsBest
from application.project import WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT,LAYER_DICT,DATA_DIR,TISSUE_TYPES_HIST,get_experiments
from application.misc.meta import COHORT_GENES_AUC,dict_label_assoc

EXPERIMENT = 'ConfounderAnalysis'
OUT_DIR = f'{OUTPUT_DIR}/{EXPERIMENT}/labels_assoc'
LABEL_ASSOC_DIR = f'{OUTPUT_DIR}/LabelCodependence/csv'
PRED_DIR =f'{OUTPUT_DIR}/Predictions'
mkdir(OUT_DIR)

experiments = get_experiments()

#Selecting model with best AUROC
testCases = selectPairsBest(
    codependence_dir=LABEL_ASSOC_DIR,
    pred_dir=PRED_DIR,
    experiments=get_experiments(),
    best_pred_genes = COHORT_GENES_AUC,
    bruns=100
)
PERM_RUNS = 100000
for cohort in ['abctb','cptac','tcga']:
    for tissueType in ['brca','colon','luad','ucec']:#TISSUE_TYPES_HIST:
        p_all_values = []
        auc_values = []
        props = []
        ORs = []
        aucs_cv = []
        positive_props = []
        run_tag = []
        pairs = testCases[tissueType]
        if cohort=='abctb' and tissueType!='brca':continue
        for v_ in pairs.keys():
            exp_tag = pairs[v_]['experiment'][0]
            cv_res = pairs[v_]['experiment'][1]
            pDf = pd.read_csv(f'{PRED_DIR}/{cohort}_{tissueType}_{exp_tag}.csv')
            strat_pairs = pairs[v_]['pairs']
            p_values = []
            for lv_ in strat_pairs:
                VOI,LV,SV = f'{lv_}',f'{v_}',f'P_{v_}' #names of the variable of interest, label and score variables
                # Dropping null value fields
                try:
                    df = pDf.loc[:,[VOI,LV,SV]].dropna()
                except:
                    print(f'Field {VOI} or {LV} not defined for this cohort')
                    continue
                voi,label,score = df[VOI], df[LV],df[SV]
                #variable of interest = voi (the variable we are intersted in analyzing e.g., the grade or CDH1 status)
                #label: true label of the predicted variable
                #score: prediction score of the predicted variable
                test_statistic = roc_auc_score#average_precision_score#
                uvals = list(set(voi))#unique values of the VOI
                N = PERM_RUNS# use at least 100000
                B = np.ones((N,len(uvals)))*np.nan
                for i in range(N):
                    pg = np.random.permutation(voi)
                    for j,v in enumerate(uvals):
                        try: #to prevent errors it cannot be computed (e.g., due to having a single class)
                            B[i,j] = test_statistic(label[pg==v],score[pg==v])
                        except:
                            continue
                # For each variable computing the AUROC and p-value
                p_val,F = [],[]
                for uIdx, v in enumerate(uvals):
                    try:
                        f = test_statistic(label[voi==v],score[voi==v])
                        p_val.append(2*min(np.mean(f>=B[:,uIdx]),np.mean(f<=B[:,uIdx])))
                        F.append(f) 
                    except:
                        print('p-values not defined for one case')
                        p_val.append(np.nan)
                        F.append(np.nan)
                print('Observed value',F)
                print('Mean/Std Background value:',np.nanmean(B,axis=0),np.nanstd(B,axis=0))
                print('p-values (for each group):',p_val)

                p_values.append(p_val)
                auc_values.append(F)
                aucs_cv.append(cv_res)
                run_tag.append(f'{v_}_{lv_}_{experiments[exp_tag]}')
                #voi_aurocs.append([max(1-roc_auc_score(voi==uvals[-1],score),roc_auc_score(voi==uvals[-1],score))]) #set value here you would like to test
                positive_props.append(df.groupby(VOI)[LV].mean().to_list())
                contTable = pd.crosstab(voi,label)
                prop = contTable.to_numpy().ravel()
                props.append(prop/prop.sum())
                contTable = contTable+1e-2
                try:
                    OR = np.log2((contTable.iloc[0,0]*contTable.iloc[1,1])/(contTable.iloc[0,1]*contTable.iloc[1,0]))
                except:
                    OR = np.nan
                # OR = 0#np.log2((contTable.iloc[0,0]*contTable.iloc[1,1])/(contTable.iloc[0,1]*contTable.iloc[1,0]))
                ORs.append([OR])
            
            if len(p_values)<=0:# No valid pair
                continue
            from statsmodels.stats.multitest import multipletests
            p_values = np.array(p_values)
            #Do the correction only when it works
            if p_values[~np.isnan(p_values).any(1)].shape[0]!=0:
                _,corp,_,_ = multipletests(
                            p_values[~np.isnan(p_values).any(1)].ravel(),
                            method='fdr_bh'
                            )
                # Updating the p-values of non-nan columns with corrected p-value
                p_values[~np.isnan(p_values).any(1)] = corp.reshape(-1,len(uvals))

            # Appending the corrected p-values to the list
            p_all_values.extend(list(p_values))
       
        pvalDf = pd.DataFrame(np.hstack((
                                        auc_values,
                                        positive_props,
                                        p_all_values,
                                        props,
                                        ORs
                                        )))
        pvalDf.index = run_tag
        pvalDf.columns = ['AUROC (WT)','AUROC (MUT)','WT +ve ratio','MUT +ve ratio','p (WT)','p (MUT)','WT-WT','WT-MUT','MUT-WT','MUT-MUT','Tendency']
        pvalDf = pvalDf.round(decimals=3)
        pvalDf.insert(loc=0,column='AUROC (STD)',value=aucs_cv)
        pvalDf.to_csv(f'{OUT_DIR}/{cohort}_{tissueType}.csv')
