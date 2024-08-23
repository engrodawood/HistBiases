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
from application.utils.performance_eval import bootstrap_runs_mout
from application.utils.statistical_tests import selectPairs,selectPairsConfounders,selectPairsConfoundersFixed
from application.utils.utilIO import mkdir
from application.utils.data_utils import load_histology
from application.misc.meta import COHORT_GENES_AUC
from application.project import WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT,LAYER_DICT,DATA_DIR,TISSUE_TYPES_HIST,get_experiments

EXPERIMENT = 'ConfounderAnalysis'
OUT_DIR = f'/{OUTPUT_DIR}/{EXPERIMENT}/TMB_HL'
PRED_DIR =f'{OUTPUT_DIR}/Predictions'
mkdir(OUT_DIR)

LABEL_ASSOC_DIR = f'{OUTPUT_DIR}/LabelCodependence/csv'

experiments = get_experiments()
#set of genes/markers predicted with higher accuracy
testCases = selectPairsConfoundersFixed(
    pred_dir=PRED_DIR,
    experiments=get_experiments(),
    best_pred_genes = COHORT_GENES_AUC,
    bruns=100
)

Flags = {'GRADE':False,'TMB':True}
PERM_RUNS = 100000
from tmb import *

thresDict = {} # defining TMB thresold only once on TCGA
for cohort in ['tcga','cptac']:#,'abctb']:
    for tissueType in TISSUE_TYPES_HIST:

        if Flags['GRADE']:
            if tissueType not in ['brca','colon','ucec']:continue
            gradeDf = load_histology(tissue=tissueType,
                                     cohort=cohort)
            gradeDf = gradeDf.loc[:,['Grade']]
            print()
        elif Flags['TMB']:
            TMB_FILE = f'{DATA_DIR}/Molecular/{cohort}/{tissueType}'
        
        # Updating the dictionary as the info is availble only for receptor status
        try:
            testCases[tissueType] = {t:testCases[tissueType][t] for t in testCases[tissueType] if t in COHORT_GENES_AUC[f'{cohort}_{tissueType}']}
        except:
            print('Something wrong')
        
        print(f'Processing {cohort} tissue type {tissueType}')
        pairs = testCases[tissueType]
        p_values = []
        auc_values = []
        props = []
        ORs = []
        aucs_cv = []
        positive_props = []
        for v_ in pairs.keys():
            exp_tag = pairs[v_][0]
            cv_res = pairs[v_][1]
            # Reading TMB from file
            try:
                df = pd.read_csv(f'{PRED_DIR}/{cohort}_{tissueType}_{exp_tag}.csv',index_col='Patient ID')
            except:
                try:
                    df = pd.read_csv(f'{PRED_DIR}/{cohort}_{tissueType}_{exp_tag}.csv',index_col='Unnamed: 0')
                except:
                    df = pd.read_csv(f'{PRED_DIR}/{cohort}_{tissueType}_{exp_tag}.csv',index_col='Identifier')
            if Flags['TMB']:
                TMBDf = getTMB(TMB_FILE,
                            ignoreGenes=[v_],
                            cohort=cohort,
                            tissueType=tissueType)
                df = df.join(TMBDf)
                thDf = df[df[f'TMB-{v_}']>0]
                #Categorizing TMB real value into 0 and 1
                if cohort in ['cptac']:
                    bin1 = thresDict[f'tcga_{tissueType}_{v_}'][0]
                    bin2 = thresDict[f'tcga_{tissueType}_{v_}'][1]
                else:
                    bin1 = np.percentile(thDf[f'TMB-{v_}'],25)
                    bin2 = np.percentile(thDf[f'TMB-{v_}'],75)
                    thresDict[f'tcga_{tissueType}_{v_}'] = [bin1,bin2]
                df.loc[df[f'TMB-{v_}']<=bin1,f'TMB-{v_}']=0
                df.loc[(df[f'TMB-{v_}']>bin1)*(df[f'TMB-{v_}']<=bin2),f'TMB-{v_}']=1
                df.loc[df[f'TMB-{v_}']>bin2,f'TMB-{v_}']=2
                
                VOI,LV,SV = f'TMB-{v_}',f'{v_}',f'P_{v_}' #names of the variable of interest, label and score variables
                df = df.loc[:,[VOI,LV,SV]].dropna()
                voi,label,score = df[VOI], df[LV],df[SV]
            elif Flags['GRADE']:
                pDf = df.join(gradeDf)
                VOI,LV,SV = 'Grade',f'{v_}',f'P_{v_}' #names of the variable of interest, label and score variables
                df = pDf.loc[:,[VOI,LV,SV]].dropna()
                voi,label,score = df[VOI], df[LV],df[SV]
            #variable of interest = voi (the variable we are intersted in analyzing e.g., the grade or CDH1 status)
            #label: true label of the predicted variable
            #score: prediction score of the predicted variable
            test_statistic = roc_auc_score#average_precision_score#
            uvals = sorted(list(set(voi)))#unique values of the VOI

            N = PERM_RUNS#100000# use at least 100000
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
        # continue
        from statsmodels.stats.multitest import multipletests
        p_values = np.array(p_values)
        auc_values = np.array(auc_values)

        pval_df_indexes = np.array(
                [f'{v_}_{experiments[testCases[tissueType][v_][0]]}'
                for v_ in testCases[tissueType].keys()]
                )
        
        if Flags['GRADE']:# In case of grade all p-values need to be corrected across all run
            _,corp,_,_ = multipletests(
                            p_values[~np.isnan(p_values).any(1)].ravel(),
                            method='fdr_bh'
                            )
            # Updating the p-values of non-nan columns with corrected p-value
            p_values[~np.isnan(p_values).any(1)] = corp.reshape(-1,len(uvals))
            pvalDf = pd.DataFrame(np.hstack((
                                    auc_values,
                                    positive_props,
                                    p_values,
                                    props,
                                    ORs
                                    )))
            pvalDf.index = pval_df_indexes#[~np.isnan(p_values).any(1)]
            if tissueType=='colon':
                pvalDf.columns = ['AUROC (L)','AUROC (H)','L ratio','H ratio','p (L)','p (H)',
                        'L-WT','L-MUT','H-WT','H-MUT','Tendency']
            else:
                pvalDf.columns = ['AUROC (L)','AUROC (M)','AUROC (H)','L ratio','M ratio','H ratio','p (L)','p (M)','p (H)',
                        'L-WT','L-MUT','M-WT','M-MUT', 'H-WT','H-MUT','Tendency']

        elif Flags['TMB']:
            # Correcting p_values in case of TMB
            for idx in range(p_values.shape[0]):
                #Selecting only non-nan values and correct the corresponding p-value
                fidx = ~np.isnan(p_values[idx,:])
                try:
                    _,corp,_,_ = multipletests(p_values[idx,fidx],
                                            method='fdr_bh'
                                            )
                    p_values[idx,fidx] = corp
                except:
                    print('Some p-values are nan')
            pvalDf = pd.DataFrame(np.hstack((
                                    auc_values,
                                    positive_props,
                                    p_values,
                                    props,
                                    ORs
                                    )))
            pvalDf.index = pval_df_indexes
            print()
            pvalDf.columns = ['AUROC (L)','AUROC (M)','AUROC (H)','L ratio','M ratio','H ratio','p (L)','p (M)','p (H)',
                        'L-WT','L-MUT','M-WT','M-MUT', 'H-WT','H-MUT','Tendency']
            # pvalDf.columns = ['AUROC (L)','AUROC (H)','L ratio','H ratio','p (L)','p (H)',
            #             'L-WT','L-MUT', 'H-WT','H-MUT','Tendency']
        pvalDf = pvalDf.round(decimals=3)
        pvalDf.insert(loc=0,column='AUROC (STD)',value=aucs_cv)
        pvalDf.to_csv(f'{OUT_DIR}/{cohort}_{tissueType}.csv')
