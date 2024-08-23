'''
Importing packages
'''

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from glob import glob
from pathlib import Path
import os
from tqdm import tqdm
import argparse

def mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=('Passing tissue name and'
            ,'worksapce dir and also graphs dir'))
    parser.add_argument('--gpu', type=str, nargs='?', const='0', default='0')
    parser.add_argument('--root_dir', type=str,
                         default='/data/PanCancer/HTEX_repo')
    parser.add_argument('--out_dir', type=str,
                         default='/data/PanCancer/HTEX_repo/CLAM/data')
    parser.add_argument('--tissue', type=str,
                         default='brca')
    parser.add_argument('--wsi_meta', type=str,
                         default='/data/PanCancer/MutPrediction/data/wsi_meta.csv')
    parser.add_argument('--feat', type=str,
                         default='SHUFFLE')
    parser.add_argument('--dthreshold', type=int,
                         default=None)
    parser.add_argument('--graphs', type=str,
                         default='/data/remote_backup/PanCan-Graphs/SHUFFLENET-20x/dth_4000')
    parser.add_argument('--nodeproba', type=bool,
    default=False)
    args = parser.parse_args()
    print('Printing args dictionary',args)
    # Saving Node Level Prediction or not
    returnNodeProba = args.nodeproba
    ROOT_DIR = args.root_dir
    OUTPUT_DIR = args.out_dir      
    GRAPHS_DIR = args.graphs
    FEAT = args.feat
    WSI_META = args.wsi_meta

    folds =5 # number of folds
    TRAIN_WHOLE = True
    CV = True

    FILTER = {'Uniq':True,'BAG':False,'OVERLAPED':False,'EX_MISSING_MPP':True}
    
      # Loading GDC Manifest and Graphs
    Manifest = pd.read_csv(f'{ROOT_DIR}/data/gdc_sample_sheet_mutation.tsv',delimiter='\t')
    Manifest.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    Manifest['Patient ID']=[Path(p).stem for p in Manifest['File Name']]
    Manifest.set_index('Patient ID',inplace=True)
    graphlist = glob(os.path.join(GRAPHS_DIR, "*.pkl"))#[0:1000]
    graphDf = pd.DataFrame(graphlist,columns=['Path'])
    graphDf['Patient ID'] = [Path(g).stem for g in graphDf['Path']]
    graphDf.set_index('Patient ID',inplace=True)

     # Excluding WSIs with missing MPP
    if FILTER['EX_MISSING_MPP']:
        metaDf = pd.read_csv(WSI_META,index_col='wsi_name')
        graphDf = graphDf.join(metaDf).dropna().loc[:,['Path']]

    mappingDf = graphDf.join(Manifest)
    projDf = mappingDf[mappingDf['Project ID']==f'TCGA-{str.upper(args.tissue)}']
    
    projDf = projDf.loc[:,['Path','Case ID']]
    
    TAG = (
            f'BAG_{FILTER["BAG"]}_overlapped_{FILTER["OVERLAPED"]}_EX_MISSING_{FILTER["EX_MISSING_MPP"]}_{args.tissue}'
             )

    
    if FILTER['Uniq']:
            selectedDf = pd.read_excel(f'{ROOT_DIR}/data/BRCA-WSI-Duplicate-Selected.xlsx',sheet_name='SELECTED-WSI').loc[:,['Patient ID','WSI Name']]
            #selectedDf = pd.read_csv(f'{ROOT_DIR}/data/selected_WSI.csv')
            projDf['WSI Name'] = [p[:23] for p in projDf.index]
            projDf = selectedDf.set_index('WSI Name').join(projDf.set_index('WSI Name'))
    # Reading Gene Group Statuses
    nGroups = 200
    targets = [i for i in range(nGroups)]
    names = ['Patient ID']+targets
    TS = pd.read_csv(f'{ROOT_DIR}/data/GroupStatuses.txt',names=names,header=None,index_col='Patient ID')
    TS.index=[t[:12] for t in TS.index]
    # Saving WSI Name as columns will be needed for generating splits for CLAM
    projDf['WSI Name'] = projDf.index.tolist()
    projDf = projDf.set_index('Case ID').join(TS).dropna()
    print(projDf.shape)
    
    # print(len(dataset),TSDf[~TSDf.index.duplicated(keep='first')].shape)
    
    if CV:
        splits_dict = {}
        for fold in range(folds):
            test_patients = set(pd.read_excel(f'{ROOT_DIR}/data/Supplementary Data.xlsx',sheet_name=f'Fold{fold+1}_Pred')['Patient ID'])
            train_patients,valid_patients = train_test_split(list(set(projDf.index)-test_patients),test_size=0.10)
            splits_dict[fold] = {'train':train_patients,'val':valid_patients,'test':list(test_patients)}
        # Generating Labels
        for voi in targets:
            label_dir =f'{OUTPUT_DIR}/G{voi}/labels'
            mkdirs(label_dir)
            labelDf = pd.DataFrame()
            labelDf['case_id'] = projDf['Patient ID'].tolist()
            labelDf['slide_id'] = projDf['WSI Name'].tolist()
            labelDf['label'] = projDf[voi].astype('int').tolist()
            labelDf.to_csv(f'{label_dir}/G{voi}_clean.csv')
        
        # Generating Splits
            splits_dir =f'{OUTPUT_DIR}/G{voi}/n_splits'
            mkdirs(splits_dir)
            for fold in range(folds):
                SplitDf = pd.DataFrame()
                train,val,test = splits_dict[fold]["train"],splits_dict[fold]["val"],splits_dict[fold]["test"]
                print(f'Gene Group: {voi} fold {fold} # train cases {len(train)} test cases {len(test)}')
                SplitDf['train'] = projDf.loc[train,'WSI Name'].tolist()
                SplitDf.loc[0:len(val)-1,'val'] = projDf.loc[val,'WSI Name'].tolist()
                SplitDf.loc[0:len(test)-1,'test'] = projDf.loc[test,'WSI Name'].tolist()
                SplitDf.to_csv(f'{splits_dir}/splits_{fold}.csv')

               


           