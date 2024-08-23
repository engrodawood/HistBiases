import pandas as pd
import numpy as np
from scipy.stats import fisher_exact,chi2_contingency
import sys
sys.path.append('.')
from application.utils.performance_eval import bootstrap_runs_mout,auroc_folds_stats
import statsmodels.stats.api as sms

def codependenceTest(df=None,
         pairs=None,
         test='fisher',
         verbose=False):
    '''
    Inputs:
        df: dataframe listing sample along rows and features or variable 
        of interest along columns
        test: statistical test to perform. Currently support ['fisher','chi-square']
        defulat is set to fisher
        pairs: list() of variables of interest [voi1, voi2] 
    Output:
        return p-value
    '''
    if not isinstance(pairs,list):
        raise ValueError('paris need to be a list see help')
    if not isinstance(df,pd.DataFrame):
        raise ValueError('df should be a dataframe')
    if not isinstance(test,str):
        raise ValueError('test should be of type string')

    v1,v2 = pairs
    # if v1=='grade' and v2=='TP53':
    #     print()
    contTable = pd.crosstab(df[v1],df[v2]); #make contingency table
    value = np.array([contTable.iloc[0].values, contTable.iloc[1].values]);
    if test=='fisher':
         result = fisher_exact(value)
    else:
        result = chi2_contingency(value)
    if verbose: 
        print('contingency table ',contTable)
        print(f'Test stat and p-value {result}')
    # Adding a psudo count to avoid division by zero
    contTable = contTable+1e-2
    odd_ratio = np.log2((contTable.iloc[0,0]*contTable.iloc[1,1])/(contTable.iloc[0,1]*contTable.iloc[1,0]))
    return result[1],odd_ratio #return only p-value

def selectPairsBest(
                codependence_dir=None,
                pred_dir = None,
                experiments=None,
                best_pred_genes=None,
                bruns = 5
                ):

    testCases = {}
    for cohort in best_pred_genes:
        if 'tcga' not in cohort:continue
        cohort_genes = best_pred_genes[cohort]
        assocDf = pd.read_csv(f'{codependence_dir}/{cohort}_fisher.csv',index_col='Unnamed: 0')
        assocDf.columns = [c.split(' ')[0] for c in assocDf.columns]
        assocDf.index = assocDf.columns
        tissueType = cohort.split('_')[1]
        testCases[tissueType] = {}
        cohort_pairs = set()
        for key in cohort_genes:
            # print(key)
            # if key=='MAP2K4':
            #     print()
            #     print()
            #The file contain *,**,*** based on p-value significane and we are selecting best one in this case
            pairs = list(set(assocDf[key].dropna().index)-{'Grade','TMB'})
            cohort_pairs = cohort_pairs.union(pairs)
            #Adding codependent pairs to testCases dictionary
            if len(pairs)>0:
                testCases[tissueType][key] = {}
                testCases[tissueType][key]['pairs'] = pairs
        # Identifying best experiment for each of these pairs based on mean auroc value across 4 CV
        cohort_pairs = list(testCases[tissueType].keys())
        aucDf = pd.DataFrame(np.zeros((len(cohort_pairs),len(experiments.keys()))))
        aucDf.index = cohort_pairs
        aucDf.columns = experiments.keys()
        from copy import deepcopy
        aucClDf = deepcopy(aucDf) 
        aucChDf = deepcopy(aucDf)   
        for exp in experiments.keys():
            df = pd.read_csv(f'{pred_dir}/{cohort}_{exp}.csv')
            aucFolds = bootstrap_runs_mout(df,targets=cohort_pairs,runs=bruns)
            CI = sms.DescrStatsW(aucFolds).tconfint_mean(alpha=0.05)
            aucDf.loc[cohort_pairs,exp] = aucFolds.mean(0)
            aucClDf.loc[cohort_pairs,exp] = CI[0]
            aucChDf.loc[cohort_pairs,exp] = CI[1]
        
        #Dropping genes predicted with AUROC below 0.60
        # aucChDf = aucChDf[aucDf.max(1)>=0.60]
        # aucClDf = aucClDf[aucDf.max(1)>=0.60]
        # aucDf = aucDf[aucDf.max(1)>=0.60]
        
        # Selecting best experiment based on AUROC
        for gene in aucDf.index:
            best_exp = aucDf.columns.tolist()[np.argmax(aucDf.loc[gene,:])]
            testCases[tissueType][gene]['experiment'] = [best_exp,
                                    f'{aucDf.loc[gene,best_exp]:.3f} ({aucClDf.loc[gene,best_exp]:.3f},{aucChDf.loc[gene,best_exp]:.3f})'
                                    ]
    return testCases

def selectPairs(
                codependence_dir=None,
                pred_dir = None,
                experiments=None,
                tissueTypes=None
                ):

    testCases = {}
    for cohort in tissueTypes:
        assocDf = pd.read_csv(f'{codependence_dir}/tcga_{cohort}_fisher.csv',index_col='Unnamed: 0')
        assocDf.columns = [c.split(' ')[0] for c in assocDf.columns]
        assocDf.index = assocDf.columns
        testCases[cohort] = {}
        cohort_pairs = set()
        for key in assocDf.columns:
            sel_col = assocDf[key]
            #The file contain *,**,*** based on p-value significane and we are selecting best one in this case
            pairs = assocDf[~sel_col.isna()].index.tolist()
            cohort_pairs = cohort_pairs.union(pairs)
            #Adding codependent pairs to testCases dictionary
            if len(pairs)>0:
                testCases[cohort][key] = pairs
        
        # Identifying best experiment for each of these pairs based on mean auroc value across 4 CV
        cohort_pairs = list(cohort_pairs)
        aucDf = pd.DataFrame(np.zeros((len(cohort_pairs),len(experiments.keys()))))
        aucDf.index = cohort_pairs
        aucDf.columns = experiments.keys()
        from copy import deepcopy
        aucStdDf = deepcopy(aucDf)   
        for exp in experiments.keys():
            df = pd.read_csv(f'{pred_dir}/tcga_{cohort}_{exp}.csv')
            #auroc = bootstrap_runs_mout(df,targets=cohort_pairs,runs=5).mean()# For first analysis runs = 100 was used but now we are reducing it
            aucFolds = auroc_folds_stats(df,targets=cohort_pairs)
            aucDf.loc[cohort_pairs,exp] = aucFolds.mean()
            aucStdDf.loc[cohort_pairs,exp] = aucFolds.std()
        
        #Dropping genes predicted with AUROC below 0.60
        aucStdDf = aucStdDf[aucDf.max(1)>=0.60]
        aucDf = aucDf[aucDf.max(1)>=0.60]
        
        gene_exp_pairs = {}
        for gene in aucDf.index:
            best_exp = aucDf.columns.tolist()[np.argmax(aucDf.loc[gene,:])]
            gene_exp_pairs[gene] = [best_exp,
                                    f'{aucDf.loc[gene,best_exp]:.3f} ({aucStdDf.loc[gene,best_exp]:.3f})'
                                    ]
        # Updating the gene pairs and adding experiment id as well
        for gene in testCases[cohort].keys():
            testCases[cohort][gene] = {k:gene_exp_pairs[k] for k in testCases[cohort][gene] if k in gene_exp_pairs.keys()}
    return testCases

def selectPairsConfounders(
                codependence_dir=None,
                pred_dir = None,
                experiments=None,
                tissueTypes=None
                ):
    testCases = {}
    for cohort in tissueTypes:
        
        #Getting list of genes from the cohort
        assocDf = pd.read_csv(f'{codependence_dir}/tcga_{cohort}_fisher.csv',index_col='Unnamed: 0')
        cohort_genes = [c.split(' ')[0] for c in assocDf.columns]

        # Slecting best experiment for each gene
        aucDf = pd.DataFrame(np.zeros((len(cohort_genes),len(experiments.keys()))))
        aucDf.index = cohort_genes
        aucDf.columns = experiments.keys()
        from copy import deepcopy
        aucStdDf = deepcopy(aucDf)   
        for exp in experiments.keys():
            df = pd.read_csv(f'{pred_dir}/tcga_{cohort}_{exp}.csv')
            aucFolds = auroc_folds_stats(df,targets=cohort_genes)
            aucDf.loc[cohort_genes,exp] = aucFolds.mean()
            aucStdDf.loc[cohort_genes,exp] = aucFolds.std()
        
        #Dropping genes predicted with AUROC below 0.60
        aucStdDf = aucStdDf[aucDf.max(1)>=0.60]
        aucDf = aucDf[aucDf.max(1)>=0.60]
        
        gene_exp_pairs = {}
        for gene in aucDf.index:
            best_exp = aucDf.columns.tolist()[np.argmax(aucDf.loc[gene,:])]
            gene_exp_pairs[gene] = [best_exp,
                                    f'{aucDf.loc[gene,best_exp]:.3f} ({aucStdDf.loc[gene,best_exp]:.3f})'
                                    ]
        testCases[cohort] = gene_exp_pairs
    return testCases

def selectPairsConfoundersFixed(
                pred_dir = None,
                experiments=None,
                best_pred_genes=None,
                bruns = 5
                ):
    testCases = {}
    for cohort in best_pred_genes:
        if 'tcga' not in cohort:continue
        cohort_genes = best_pred_genes[cohort]

        # Slecting best experiment for each gene
        aucDf = pd.DataFrame(np.zeros((len(cohort_genes),len(experiments.keys()))))
        aucDf.index = cohort_genes
        aucDf.columns = experiments.keys()
        from copy import deepcopy
        aucClDf = deepcopy(aucDf) 
        aucChDf = deepcopy(aucDf)   
        for exp in experiments.keys():
            df = pd.read_csv(f'{pred_dir}/{cohort}_{exp}.csv')
            aucFolds = bootstrap_runs_mout(df,targets=cohort_genes,runs=bruns)
            CI = sms.DescrStatsW(aucFolds).tconfint_mean(alpha=0.05)
            aucDf.loc[cohort_genes,exp] = aucFolds.mean(0)
            aucClDf.loc[cohort_genes,exp] = CI[0]
            aucChDf.loc[cohort_genes,exp] = CI[1]
        
        #Dropping genes predicted with AUROC below 0.60
        aucChDf = aucChDf[aucDf.max(1)>=0.60]
        aucClDf = aucClDf[aucDf.max(1)>=0.60]
        aucDf = aucDf[aucDf.max(1)>=0.60]
        
        gene_exp_pairs = {}
        for gene in aucDf.index:
            best_exp = aucDf.columns.tolist()[np.argmax(aucDf.loc[gene,:])]
            gene_exp_pairs[gene] = [best_exp,
                                    f'{aucDf.loc[gene,best_exp]:.3f} ({aucClDf.loc[gene,best_exp]:.3f},{aucChDf.loc[gene,best_exp]:.3f})'
                                    ]
        testCases[cohort.split('_')[-1]] = gene_exp_pairs
    return testCases

if __name__=='__main__':
    print()
    import sys
    sys.path.append('.')
    from application.utils.performance_eval import bootstrap_runs_mout
    from application.utils.statistical_tests import selectPairs,selectPairsConfounders
    from application.utils.utilIO import mkdir
    from application.utils.data_utils import load_histology
    from application.misc.meta import COHORT_GENES_AUC
    from application.project import WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT,LAYER_DICT,DATA_DIR,TISSUE_TYPES_HIST,get_experiments
    testCases = selectPairsConfoundersFixed(
        pred_dir=f'{OUTPUT_DIR}/Predictions',
        experiments=get_experiments(),
        best_pred_genes = COHORT_GENES_AUC
    )
    print()