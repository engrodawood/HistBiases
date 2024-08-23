from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import torch.nn.functional as F
from glob import glob
import os
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')
from application.project import WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT,LAYER_DICT
from application.utils.utilIO import mkdir
from application.utils.data_utils import load_manifest,load_meta
FILTER = {'BAG':False,'EX_MISSING_MPP':True}
if __name__ == '__main__':

    for Repr in CLAM_DICT.keys():
        EXP_TAG = f'FEAT_{Repr}_lr_0.001_decay_0.0001_bsize_8_layers_{LAYER_DICT[Repr]}_dth_None_convEdgeConvBAG_True_overlapped_False_EX_MISSING_True'
        FEAT_DIR = CLAM_DICT[Repr]['IN_FEATS_DIR']
        BAGS_OUT_DIR = CLAM_DICT[Repr]['OUT_BAGS_DIR'] 
        OUT_SPLITS_DIR = CLAM_DICT[Repr]['OUT_SPLITS_DIR']
        #Emplying same selection criteria inplace for WSIs
        # Loading GDC Manifest and Graphs
        manifestDf = load_manifest() 
        featsList = [f for f in glob(os.path.join(FEAT_DIR, "*_feat.npy"))]
        featDf = pd.DataFrame(featsList,columns=['Path'])
        featDf['Patient ID'] = [Path(g).stem[:-5] for g in featDf['Path']]
        featDf.set_index('Patient ID',inplace=True)

        # Excluding WSIs with missing MPP
        if FILTER['EX_MISSING_MPP']:
            metaDf = load_meta()
            featDf = featDf.join(metaDf).dropna().loc[:,['Path']]

        mappingDf = featDf.join(manifestDf)
        COHORT = 'tcga'
        FOLDS_DIR = f'{OUTPUT_DIR}/HistPred/{COHORT}/SlideGraph/foldPred'
        #EXP_TAG = 'lr_0.001_decay_0.0001_bsize_8_layers_1024_1024_1024_dth_None_convEdgeConvBAG_True_overlapped_False_EX_MISSING_True'
        #EXP_TAG = 'FEAT_CTransPath_lr_0.001_decay_0.0001_bsize_8_layers_768_768_768_dth_None_convEdgeConvBAG_True_overlapped_False_EX_MISSING_True'
        for TISSUE in ['colon']:#['ucec','brca','colon','luad']:
            projDf = mappingDf[mappingDf['Project ID']==f'TCGA-{str.upper(TISSUE)}'].dropna()
            projDf.index = [p[:12] for p in projDf.index]
            
            #Saving feature bag if we have multiple slides for a sigle case
            if FILTER['BAG']:
                BAGS_DIR = f'{BAGS_OUT_DIR}/{TISSUE}'
                print(BAGS_DIR)
                mkdir(BAGS_DIR)
                patientIDs = set(projDf.index.tolist())
                for pid in patientIDs:
                    wsis = projDf[projDf.Path.str.contains(pid)].Path.tolist()
                    bagFeats = []
                    for pth in wsis:
                        F = np.load(pth)
                        bagFeats.extend(list(F))
                    np.save(f'{BAGS_DIR}/{pid}_feat.npy',np.array(bagFeats))
                    
            PRED_DIR = f'{FOLDS_DIR}/{EXP_TAG}{TISSUE}'
            geneList = os.listdir(PRED_DIR)
            geneList = ['CINGS']#,'HM','CIMP']
            for gene in geneList:   
                foldFiles = os.listdir(f'{PRED_DIR}/{gene}')
                gtDf = pd.DataFrame()
                splits_dict = {}
                for foldFile in foldFiles:
                    foldDf = pd.read_csv(f'{PRED_DIR}/{gene}/{foldFile}',index_col='Patient ID')
                    foldDf['fold_idx'] = int(foldFile.split('.')[0])
                    if gtDf.shape[0]==0:
                        gtDf = foldDf
                    else:
                        gtDf = pd.concat([gtDf,foldDf])
                # Selecting train and test patinets based on fold ids
                n_folds = 4
                splits_dir =f'{OUT_SPLITS_DIR}/{TISSUE}/{gene}/n_splits'
                label_dir =f'{OUT_SPLITS_DIR}/{TISSUE}/{gene}/labels'
                mkdir(splits_dir)
                mkdir(label_dir)
                for fold in range(n_folds):
                    SplitDf = pd.DataFrame()
                    test_patients = set(gtDf[gtDf['fold_idx']==fold].index)
                    train_val_patients = list(set(gtDf.index)-test_patients)
                    train_patients,valid_patients = train_test_split(train_val_patients,test_size=0.10,
                                                                    stratify=gtDf.loc[train_val_patients,f'T_{gene}'].tolist())
                    print(f'Gene: {gene} fold {fold} # train cases {len(train_patients)} test cases {len(valid_patients)}')
                    print(np.unique(gtDf.loc[valid_patients,f'T_{gene}'],return_counts=True))
                    SplitDf['train'] = train_patients
                    SplitDf.loc[0:len(valid_patients)-1,'val'] = valid_patients
                    SplitDf.loc[0:len(test_patients)-1,'test'] = list(test_patients)
                    SplitDf.to_csv(f'{splits_dir}/splits_{fold}.csv')
                #Label dataframe
                labelDf = pd.DataFrame()
                labelDf['case_id'] = gtDf.index.tolist()
                labelDf['slide_id'] = gtDf.index.tolist()
                labelDf['label'] = gtDf.loc[:,f'T_{gene}'].tolist()
                labelDf.to_csv(f'{label_dir}/{gene}_clean.csv')
            

