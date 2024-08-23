'''
Importing packages
'''

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR,CLAM_DICT_EXTRV_CPTAC,TISSUE_TYPES_HIST
from application.utils.data_utils import load_molecular
from application.utils.utilIO import mkdir
from pathlib import Path

TRAIN_COHORT = 'tcga'
INFER_COHORT = 'cptac'
device = 'cuda:1'

cptac_mapping = {
    'colon':'coad',
    'brca':'brca',
    'luad':'luad',
    'ucec':'ucec'
}

def preproc_cptac(
        featDf=None,
        tissue='brca'
        ):
    if tissue in ['LUAD','UCEC']:
        featDf.loc[:,'Patient ID'] = [Path(p).stem.replace(f'{tissue.upper()}_','')[:9] for p in featDf['Path'].tolist()]
    elif tissue in ['abctb']:# this is not a tissue just to put it in contex
        featDf.loc[:,'Patient ID'] = ['-'.join(Path(g).stem.split('-')[2:]).split('.')[0] for g in featDf['Path'].tolist()]#[0:50]
    else:
        featDf.loc[:,'Patient ID'] = [Path(p).stem.split('-')[0].replace(f'{tissue.upper()}_','') for p in featDf['Path'].tolist()]
    featDf.set_index('Patient ID',inplace=True)
    return featDf

for Repr in CLAM_DICT_EXTRV_CPTAC.keys():
    IN_FEATS_DIR = CLAM_DICT_EXTRV_CPTAC[Repr]['IN_FEATS_DIR']
    OUT_BAGS_DIR = CLAM_DICT_EXTRV_CPTAC[Repr]['OUT_BAGS_DIR']
    OUT_SPLITS_DIR = CLAM_DICT_EXTRV_CPTAC[Repr]['OUT_SPLITS_DIR']
    for tissue in TISSUE_TYPES_HIST:
        #Loading true marker status for these patients
        npyDf = pd.DataFrame()
        BioDf = load_molecular(cohort=INFER_COHORT,tissue=tissue)
        if tissue in ['brca']:
            BioDf.index = [idx[1:] for idx in BioDf.index]
        
        # Selecting set of WSI features
        npyDf = pd.DataFrame([g for g in glob(os.path.join(IN_FEATS_DIR, "*feat.npy")) if f'{cptac_mapping[tissue].upper()}_' in g],
                             columns = ['Path'])
        featDf = preproc_cptac(featDf=npyDf,
                                   tissue=cptac_mapping[tissue].upper())

        # Removing duplicated from the target file
        fBDf = featDf.join(BioDf)
        fBDf = fBDf[~fBDf.index.duplicated()]
        splits_dict = {}
        #Train test and val are same
        folds = CLAM_DICT_EXTRV_CPTAC[Repr]['folds']
         # Generating Labels
        targets = list(set(BioDf.columns.tolist()))
       
        for voi in targets:
            selDf = fBDf.loc[:,[voi]].dropna()
            labelPids = selDf.index.tolist() 
            labelDf = pd.DataFrame()
            labelDf['case_id'] = labelPids
            labelDf['slide_id'] = labelPids
            labelDf['label'] = selDf.loc[labelPids,voi].astype('int').tolist()
            voi = voi.replace(' status','') # avoiding space 
            label_dir =f'{OUT_SPLITS_DIR}/{tissue}/{voi}/labels'
            mkdir(label_dir)
            labelDf.to_csv(f'{label_dir}/{voi}_clean.csv')
        
        # Generating Splits
            splits_dir =label_dir =f'{OUT_SPLITS_DIR}/{tissue}/{voi}/n_splits'
            mkdir(splits_dir)
            for fold in range(folds):
                splits_dict[fold] = {'train':labelPids,'val':labelPids,'test':labelPids}
                SplitDf = pd.DataFrame()
                train,val,test = splits_dict[fold]["train"],splits_dict[fold]["val"],splits_dict[fold]["test"]
                print(f'Biomarker: {voi} fold {fold} # train cases {len(train)} test cases {len(test)}')
                SplitDf['train'] = labelPids
                SplitDf.loc[0:len(val)-1,'val'] = labelPids
                SplitDf.loc[0:len(test)-1,'test'] = labelPids
                SplitDf.to_csv(f'{splits_dir}/splits_{fold}.csv')
        
        # Saving all bags at once for each cohort
        continue
        mapDf = featDf.join(BioDf)
        patientIDs = set(mapDf.index.tolist())
        from tqdm import tqdm
        mkdir(OUT_BAGS_DIR)
        for pid in tqdm(patientIDs):
            wsis = mapDf[mapDf.Path.str.contains(pid)].Path.tolist()
            bagFeats = []
            import numpy as np
            for pth in wsis:
                F = np.load(pth)
                bagFeats.extend(list(F))
            np.save(f'{OUT_BAGS_DIR}/{pid}_feat.npy',np.array(bagFeats))
        #Loading graphs for each cohort
        