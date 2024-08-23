# Changing to parent directory for acessing the application folder
import sys
import os
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR
import numpy as np
from application.utils.utilIO import mkdir
import pandas as pd
from application.misc.meta import dict_map_msi,dict_map_receptor
from pathlib import Path

def load_meta(cohort='tcga'):
    metaDf = pd.read_csv(f'{DATA_DIR}/META_WSIs/wsi_meta.csv',index_col='wsi_name')
    return metaDf

def load_manifest(cohort='tcga'):
    Manifest = pd.read_csv(f'{DATA_DIR}/META_WSIs/gdc_sample_sheet_mutation.tsv',delimiter='\t')
    Manifest.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    Manifest['Patient ID']=[Path(p).stem for p in Manifest['File Name']]
    Manifest.set_index('Patient ID',inplace=True)
    return Manifest


def get_tmb(in_path):
    CD = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
    cols_to_filter = ['TMB_NONSYNONYMOUS']
    CD = CD.loc[:,cols_to_filter]
    CD.rename(columns={'TMB_NONSYNONYMOUS':'TMB'},inplace=True)
    #Binarizing TMB
    med_tmb = 10#CD['TMB'].median()
    CD[CD.TMB<=med_tmb] = 0
    CD[CD.TMB>med_tmb] = 1
    return CD
def load_histology(cohort='tcga',
                   tissue='brca'):
        if tissue in ['brca']:
            if cohort=='abctb':
                HIST = pd.read_excel('/data/PanCancer/HTEX_repo/2021_1793_Fayyaz_Path_Data_Final_05-08-2022.xlsx',sheet_name='Pathology Data',index_col='Identifier')
                field = 'Histopathological Grade'
                #HSPLIT = HIST['Histopathological Grade'].str.split().str.len()# for preproc
                HIST[field].replace({'No grade':np.nan,'Nuclear grade 2':'Inv. carc. grade 2'},inplace=True)
                #Dropping patients with no grade information
                HIST = HIST[[field]].dropna()
                HIST['patient_id'] = HIST.index.tolist()
                gradeField = HIST[field].str.split().to_list()
                HIST['Grade'] = [int(g[-1]) for g in gradeField]
                #Picking only patients with a single grade
                HIST = HIST[HIST.groupby('patient_id')['Grade'].nunique()<=1]
                HIST.drop(columns=[field,'patient_id'],inplace=True)
                return HIST
            elif cohort=='tcga':
                hist_feats = [
                'Epithelial area',
                '2016 Histology Annotations',
                'Inflammation', 'LCIS', 'Apocrine features', 'DCIS',
                'Epithelial tubule formation', 'Lymphovascular Invasion (LVI)',
                'Necrosis', 'Nuclear pleomorphism', 'Fibrous focus', 'Mitosis'
                 ]

                # Reading the histology data
                HData = pd.read_excel(f'{DATA_DIR}/Histology/{cohort}_{tissue}/NIHMS1772917-supplement-Supplementary_Data_S2.xlsx')
                HData.set_index('Sample CLID',inplace=True)
                HData = HData.loc[:,hist_feats]
                histCol = '2016 Histology Annotations'
                # import pdb; pdb.set_trace()
                for col in HData.columns:
                    HData.loc[:,col]=HData.loc[:,col].str.strip()
                FHistPatterns = {key:HData.loc[:,histCol].tolist().count(key) for key in set(HData.loc[:,histCol])
                                        if HData.loc[:,histCol].tolist().count(key)>1 and key!='Not available'}

                histTypes = True
                if histTypes:
                    # Generate one host encoding of the features. 
                    for hpat in FHistPatterns:
                        pat_subset = HData[HData[histCol]==hpat].index.tolist()
                        HData.loc[:,hpat] = 0
                        HData.loc[pat_subset,hpat] = 1
                HData.drop(columns=[histCol],inplace=True)

                '''
                Removing spaces between entries in columns
                '''
                featMapping = {
                            'Epithelial area':{'<25% (Low)':0,'25-75% (Moderate)':1,'>75% (High)':2,np.nan:np.nan},
                            'Inflammation':{'Absent':0,'Present':1,np.nan:np.nan},
                            'DCIS':{'Absent':0, 'Present':1, np.nan:np.nan},
                            'LCIS':{'Absent':0, 'Present':1, np.nan:np.nan},
                            'Apocrine features':{'Absent':0,'1-5% (Minimum)':1, '6-50% (Moderate)':2, '>50% (Marked)':3, np.nan:np.nan},
                            'Epithelial tubule formation':{'(score = 1) >75%':1, '(score = 2) 10% to 75%':2, '(score = 3) <10%':3, np.nan:np.nan},
                            'Necrosis':{'Absent':0, 'Present':1, np.nan:np.nan},

                            'Lymphovascular Invasion (LVI)':{'Absent':0,'Cannot be evaluated.':np.nan,'Frequent':2,'Present':1,
                            'No non-tumor tissue is present - cannot be evaluated.':np.nan,
                            'No non-tumor tissue is present for evaluation.':np.nan,
                            'There is an area of central fibrosis of lower cellularity, but not a typical completely fibrotic center.':np.nan,
                            np.nan:np.nan
                            },

                            'Nuclear pleomorphism':{'(score = 1) Small regular nuclei':1,
                            '(score = 2) Moderate increase in size':2,
                            '(score = 3) Marked variation in size, prominent nucleoli, chromatin clumping':3,
                            np.nan:np.nan
                            },

                            'Mitosis':{'(score = 1) 0 to 5 per 10 HPF':1,
                            '(score = 2) 6 to 10 per 10 HPF':2,
                            '(score = 3) >10 per 10 HPF':3,
                            np.nan:np.nan
                            },
                            'Fibrous focus':{'Absent':0, 'Cannot be evaluated.':np.nan, 'Multiple Fibrotic Foci':1, np.nan:np.nan}

                            }

                # import pdb; pdb.set_trace()
                HData.replace(featMapping,inplace=True)

                HGrade = HData.loc[:,['Epithelial tubule formation','Nuclear pleomorphism','Mitosis']].dropna()
                HGrade.loc[HGrade.index,'n_grade'] = HGrade.sum(1)
                HData.loc[HGrade.index,'Grade'] = HGrade.loc[HGrade.index,'n_grade']
                g3_idx = HData['Grade']>=8
                g2_idx = (HData['Grade']>=6)*(HData['Grade']<=7)
                g1_idx = (HData['Grade']>=3)*(HData['Grade']<=5)
                HData.loc[g3_idx,'Grade']=3
                HData.loc[g2_idx,'Grade']=2
                HData.loc[g1_idx,'Grade']=1

                # Reading TILS regional Fraction from Saltz et al paper
                immune_file = f'{DATA_DIR}/Histology/{cohort}_{tissue}/NIHMS958212-supplement-2.xlsx'
                #'Lymphocytes','Plasma Cells'
                cols2read = ['TIL Regional Fraction']

                TILs = pd.read_excel(immune_file).rename(columns = {'TCGA Participant Barcode':'Patient ID'}).set_index('Patient ID')
                CCHData = HData.join(TILs[TILs['TCGA Study']=='BRCA'].loc[:,cols2read])
                return CCHData.loc[:,['Grade']]
            elif cohort=='metabric':
                in_path = f'{DATA_DIR}/Molecular/{cohort}_{tissue}/data_clinical_sample.txt'
                gradeDf = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
                gradeDf = gradeDf.loc[:,['GRADE']]
                gradeDf.rename(columns={'GRADE':'Grade'},inplace=True)
                return gradeDf                

        elif tissue in ['colon']:
            if cohort=='tcga':
                gradeDf = pd.read_excel(f'{DATA_DIR}/Histology/{cohort}_{tissue}/TCGA_COAD_Grades (1).xlsx',index_col = 'Patient ID')
                gradeDf.rename(columns={'Histologic grade':'Grade'},inplace=True)
                return gradeDf
            elif cohort=='dfci':
                in_path = f'{DATA_DIR}/Molecular/{cohort}_{tissue}/data_clinical_sample.txt'
                gradeDf = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
                cols_to_filter = ['PRIMARY_TUMOR_GRADE']
                gradeDf = gradeDf.loc[:,cols_to_filter]
                preproc = {'Well-Moderate':0,'Poor':1}
                gradeDf.replace(preproc,inplace=True)
                gradeDf.rename(columns={'PRIMARY_TUMOR_GRADE':'Grade'},inplace=True)
                return gradeDf
        elif tissue in ['ucec']:
            if cohort=='tcga':
                in_path = f'{DATA_DIR}/Molecular/TMB/data_{cohort}_{tissue}_sample.txt'
                gradeDf = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
                cols_to_filter = ['GRADE']
                gradeDf = gradeDf.loc[:,cols_to_filter]
                preproc = {'High Grade':3,'G1':1,'G2':2,'G3':3}
                gradeDf.replace(preproc,inplace=True)
                gradeDf.rename(columns={'GRADE':'Grade'},inplace=True)
                return gradeDf
            elif cohort=='cptac':
                in_path = f'{DATA_DIR}/Histology/cptac_ucec/mmc2 cptac.xlsx'
                gradeDf = pd.read_excel(in_path,sheet_name='Supplementary Table 1', index_col='Case_ID')
                gradeDf = gradeDf[gradeDf.tumor_code==tissue.upper()]
                cols_to_filter = ['cptac_path/histologic_grade']
                gradeDf = gradeDf.loc[:,cols_to_filter]
                preproc = {'G1 Well differentiated':1, 'G2 Moderately differentiated':2, 'G3 Poorly differentiated':3, 'GX Grading is not applicable, cannot be assessed or not specified':np.nan}
                gradeDf.replace(preproc,inplace=True)
                gradeDf.rename(columns={'cptac_path/histologic_grade':'Grade'},inplace=True)
                return gradeDf


def onhot_encoder(df=None,columns=None):
    for field in columns:
        for imType in set(df[field].dropna().tolist()):
        # import pdb; pdb.set_trace()
            tempDf = df[field].dropna()
            tempDf[tempDf!=imType]=0
            tempDf[tempDf==imType]=1
            df.loc[tempDf.index,f'{field}_{imType}'] =tempDf 
    df.drop(columns=columns,inplace=True)
    return df
     
def load_molecular(cohort='tcga',tissue='brca'):
    if cohort=='abctb':
        MUT = pd.read_excel('/data/PanCancer/HTEX_repo/2021_1793_Fayyaz_Path_Data_Final_05-08-2022.xlsx',sheet_name='Pathology Data',index_col='Identifier')
        columns = ['ER Result','PR Result','HER2 Result']
        MUT = MUT.loc[:,columns]
        MUT = MUT[~MUT.index.duplicated()]
        MUT.rename(columns = {'ER Result':'ER status','PR Result':'PR status','HER2 Result':'HER2 status'},inplace=True)
        MUT.replace({'Negative':0,'negative':0,'Positive':1,'positive':1,'Not performed':np.nan,'Positive ':1,'Equivocal':np.nan,'Negative ':np.nan,'Not reported':np.nan},inplace=True)
        return MUT
       
    # #Loading mutation data
    MUT = pd.read_csv(f'{DATA_DIR}/MUT/{cohort}_{tissue}_b.csv',index_col='Patient ID')
    #Getting tmB
    TMB_FILE = f'{DATA_DIR}/Molecular/TMB/data_{cohort}_{tissue}_sample.txt'
    TMB = get_tmb(TMB_FILE)
    if cohort in ['tcga']:
        TMB.index = [t[:12] for t in TMB.index]
        MUT.index = [t[:12] for t in MUT.index]
    if cohort in ['msk'] and tissue in ['luad']:
        MUT.index = ['-'.join(m.split('-')[:2]) for m in MUT.index]
    MUT = MUT.join(TMB)
    if tissue=='brca':
        if cohort=='metabric':
            in_path = f'{DATA_DIR}/Molecular/{cohort}_{tissue}/data_clinical_sample.txt'
            CD = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
            cols_to_filter = ['ER_STATUS', 'PR_STATUS','HER2_STATUS']
            CD = CD.loc[:,cols_to_filter]
            CD.rename(columns=dict(zip(cols_to_filter,['ER status','PR status','HER2 status'])),inplace=True)
            CD.replace(dict_map_receptor,inplace=True)
            return MUT.join(CD)
        elif cohort=='cptac':
            in_path = f'{DATA_DIR}/Molecular/{cohort}/{tissue}_data_clinical_patient.txt'
            CD = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
            cols_to_filter = ['ER_UPDATED_CLINICAL_STATUS','PR_CLINICAL_STATUS','ERBB2_UPDATED_CLINICAL_STATUS']
            CD = CD.loc[:,cols_to_filter]
            CD.rename(columns=dict(zip(cols_to_filter,['ER status','PR status','HER2 status'])),inplace=True)
            CD.replace(dict_map_receptor,inplace=True)
            return MUT.join(CD)

        else:
            in_path = f'{DATA_DIR}/Molecular/{cohort}_{tissue}/nationwidechildrens.org_clinical_patient_brca (4).txt'

            #second and third row contain some extra information so exclude that
            CD = pd.read_csv(in_path,delimiter='\t',skiprows=range(1,3),index_col='bcr_patient_barcode')

            cols_to_filter = ['er_status_by_ihc', 'pr_status_by_ihc','her2_status_by_ihc']
            CD = CD.loc[:,cols_to_filter]
            CD.rename(columns = {
                                'er_status_by_ihc':'ER status',
                                'pr_status_by_ihc':'PR status',
                                'her2_status_by_ihc':'HER2 status'
                                },
                                inplace=True)

            # Approach for binarization
            CD.replace(dict_map_receptor,inplace=True)
            MUT.index = [idx[:12] for idx in MUT.index]
            return MUT.join(CD)
    elif tissue=='colon':
        if cohort=='tcga':
            CB = pd.read_excel(f'{DATA_DIR}/Molecular/{cohort}_{tissue}/idars_mmc2.xlsx',sheet_name='TCGA_CRC_GroundTruths',index_col='PATIENT')
            CBcols = {'Hypermutated':'HM','CINGSBinaryClassification':'CINGS','HypermethylationCategory':'CIMP'}
            CB = CB.loc[:,list(CBcols.keys())]
            #Necessery pre-processing
            CB.rename(columns = CBcols,inplace=True)
            preproc = {'CRC CIMP-L':0,'GEA CIMP-L':0,'Non-CIMP':0,'CIMP-H':1}
            CB.replace(preproc,inplace=True)
            MUT.index = [idx[:12] for idx in MUT.index]
            CD = pd.read_csv(f'{DATA_DIR}/Molecular/{cohort}_{tissue}/TCGA_622PTs_iCMS_and_CMS_label (1) (1).csv',index_col ='patient_id')
            CD.replace(dict_map_msi,inplace=True)
            CD = CD[~CD.index.duplicated()]
            encode_onehot = ['CMS','iCMS']
            select_columns = ['MSI.status']+encode_onehot
            CD = CD.loc[:,select_columns]
            CD = onhot_encoder(df = CD,
                                columns = encode_onehot)
            CD.rename(columns = {'MSI.status':'MSI status'},inplace=True)
            
            return CD.join(MUT).join(CB)
        
        elif cohort=='dfci':
            in_path = f'{DATA_DIR}/Molecular/{cohort}_{tissue}/data_clinical_sample.txt'
            CD = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
            cols_to_filter = ['MSI_STATUS','CIMP_CATEGORY']
            CD = CD.loc[:,cols_to_filter]
            preproc = {'CIMP-high':1,'CIMP-0/low':0}
            CD.replace(preproc,inplace=True)
            CD.rename(columns=dict(zip(cols_to_filter,['MSI status','CIMP'])),inplace=True)
            CD.replace(dict_map_msi,inplace=True)
            return MUT.join(CD)
        elif cohort=='cptac':
            in_path = f'{DATA_DIR}/Molecular/{cohort}/{tissue}_data_clinical_patient.txt'
            CD = pd.read_csv(in_path,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
            cols_to_filter = ['MUTATION_PHENOTYPE']
            CD = CD.loc[:,cols_to_filter]
            CD.rename(columns=dict(zip(cols_to_filter,['MSI status'])),inplace=True)
            CD.replace(dict_map_msi,inplace=True)
            return MUT.join(CD)
    else:
        return MUT

if __name__=='__main__':
    mutDf = load_molecular(tissue='brca',cohort='abctb_dup')
    print()
    print()
    #histDf = load_histology(tissue='colon')
