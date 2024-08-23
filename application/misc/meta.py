import numpy as np

COHORT_GENES = {
    'brca':'AKT1, ARID1A, CDH1, GATA3, KMT2C, MAP2K4, MAP3K1, NCOR1, PIK3CA, PTEN, RUNX1, TP53, NF1',
    'colon':'AMER1, APC, ARID1A, ATM, BRAF, FBXW7, KRAS, NRAS, PIK3CA, PIK3R1, PTEN, RNF43, SMAD4, SOX9, TCF7L2, TP53',
    'gbm':'ATRX, EGFR, IDH1, NF1, PIK3CA, PIK3R1, PTEN, RB1, TP53',
    'luad':'BRAF, EGFR, KEAP1, KRAS, MGA, NF1, RBM10, SMARCA4, STK11, TP53',
    'lusc':'CDKN2A, FAT1, KMT2D, NF1, NFE2L2, PIK3CA, PTEN, RB1, TP53',
    'paad':'CDKN2A, KRAS, SMAD4, TP53',
    'ucec':('KMT2C, APC, ARID1A, ARID4B, ARID5B, ATM, ATRX, BCOR, BCORL1, BRCA2, CASP8, CTCF,'
         'CTNNB1, CUX1, DICER1, EP300, FAT1, FBXW7, FGFR2, INPPL1, JAK1, KMT2A, KMT2B, KMT2D, KRAS, MAP3K1, MGA,'
        'MSH6, NF1, NSD1, PIK3CA, PIK3R1, POLE, PPP2R1A, PTCH1, PTEN, RASA1, RB1, RNF43, SETD2, SPOP, TP53, ZFHX3')
}

COHORT_GENES_AUC = {
    'tcga_brca':['ER', 'PR', 'HER2', 'CDH1', 'TP53', 'MAP3K1', 'PIK3CA', 'ARID1A'],
    'cptac_brca':['ER', 'PR', 'HER2', 'CDH1', 'TP53', 'MAP3K1', 'PIK3CA', 'ARID1A'],
    'tcga_colon':['MSI', 'CIMP', 'CINGS', 'HM','BRAF','TP53','RNF43','APC', 'ATM', 'SOX9','KRAS','PIK3CA'],
    'cptac_colon':['MSI','BRAF','TP53','RNF43','APC', 'ATM', 'SOX9','KRAS','PIK3CA'],
    'abctb_brca':['ER', 'PR', 'HER2'],
    'tcga_ucec':['PTEN', 'TP53', 'CTNNB1', 'ARID1A', 'KRAS', 'MGA', 'FGFR2', 'CTCF', 'JAK1', 'NSD1', 'APC', 'RNF43'],
    'cptac_ucec':['PTEN', 'TP53', 'CTNNB1', 'ARID1A', 'KRAS', 'MGA', 'FGFR2', 'CTCF', 'JAK1', 'NSD1', 'APC', 'RNF43'],
    'tcga_luad':['TP53', 'EGFR', 'KEAP1', 'STK11','SMARCA4', 'KRAS'],
    'cptac_luad':['TP53', 'EGFR', 'KEAP1', 'STK11','SMARCA4', 'KRAS']

}

COHORT_GENES_TMB = {
    'tcga_brca':['ER', 'PR', 'TP53', 'CDH1','MAP3K1', 'PIK3CA'],
    'cptac_brca':['ER', 'PR', 'TP53','CDH1', 'MAP3K1', 'PIK3CA'],
    'tcga_colon':['MSI','BRAF','TP53','APC', 'SOX9','KRAS','PIK3CA'],
    'cptac_colon':['MSI','BRAF','TP53','APC', 'SOX9','KRAS','PIK3CA'],
    'abctb_brca':['ER', 'PR', 'HER2'],
    'tcga_ucec':['PTEN', 'TP53', 'CTNNB1', 'ARID1A', 'KRAS', 'MGA', 'FGFR2', 'CTCF', 'JAK1', 'NSD1', 'APC', 'RNF43'],
    'cptac_ucec':['PTEN', 'TP53', 'CTNNB1', 'ARID1A', 'KRAS', 'MGA', 'FGFR2', 'CTCF', 'JAK1', 'NSD1', 'APC', 'RNF43'],
    'tcga_luad':['TP53', 'EGFR', 'KEAP1', 'STK11','SMARCA4', 'KRAS'],
    'cptac_luad':['TP53', 'EGFR', 'KEAP1', 'STK11','SMARCA4', 'KRAS']

}

MOL_MARKERS_CODEP = {
    'tcga_colon':['MSI status','CIMP','CINGS','HM','TMB'],
    'tcga_luad':['TMB'],
    'tcga_ucec':['TMB'],
    'dfci_colon':['MSI status','CIMP','TMB'],
    'brca':['ER status','PR status','HER2 status','TMB'],
    'tcga_brca':['ER status','PR status','HER2 status','TMB'],
    'metabric_brca':['ER status','PR status','HER2 status','TMB'],
    'msk_ucec':['TMB'],
    'msk_luad':['TMB']
}


dict_map_msi = {'MSI-H':1.0,'MSI-high':1.0,'MSI-low':0.0,'MSI-L':0.0,'MSS':0.0,'[Not Available]':np.nan}
dict_map_receptor = {
                'Positive':1,'Negative':0,'Indeterminate':np.nan,'[Not Evaluated]':np.nan,
                'Equivocal':np.nan, '[Not Available]':np.nan,
                'negative':0,'positive':1,
                'equivocal':np.nan,'Not reported':np.nan,'Not performed':np.nan
            }


dict_label_assoc = {

    'tcga':{
        'colon':{
                'MSI':['HM','CIMP','CINGS','BRAF','APC'],
                'CIMP':['HM','CINGS','APC'],
                'CINGS':['APC','HM'],
                'HM':['ATM','APC'],
                'BRAF':['HM','CIMP','CINGS','MSI','APC','RNF43'],
                #'APC':['HM','CIMP','CINGS','MSI','TP53','BRAF'],
                'RNF43':['MSI','BRAF','APC'],
                    
                    },
        'brca':{
                'ER':['PR','TP53','CDH1','GATA3','PIK3CA'],
                'PR':['ER','TP53','CDH1'],
                'TP53':['ER','PR','CDH1','MAP3K1'],
                'CDH1':['ER'],
                'PIK3CA':['ER','PR']

        },
        
        'ucec':{
            'PTEN':['BRCA2','CTNNB1','CTCF','APC','RNF43','TP53'],
            'TP53':['KRAS','ARID1A','RNF43','PTEN','CTCF'],
            'CTNNB1':['MGA','ARID1A','TP53','PTEN'],
            'ARID1A':['TP53','CTCF','PTEN']
        },
        
        'luad':{'KRAS':['STK11','NF1'],
                'TP53':['STK11'],
        },
                    },
    'cptac':{
        'colon':{
                'MSI':['APC','BRAF'],
                'BRAF':['MSI','APC','RNF43'],
                'APC':['MSI','TP53','BRAF'],
                'RNF43':['MSI','BRAF','APC']
                      },
        'brca':{
            'ER':['PR','TP53','CDH1','GATA3','PIK3CA'],
            'PR':['ER','TP53','CDH1'],
            'TP53':['ER','PR','CDH1','MAP3K1'],
            'CDH1':['ER'],
            'PIK3CA':['ER','PR']

        },
         'ucec':{
            'PTEN':['BRCA2','CTNNB1','CTCF','APC','RNF43','TP53'],
            'TP53':['KRAS','ARID1A','RNF43','PTEN','CTCF'],
            'CTNNB1':['MGA','ARID1A','TP53','PTEN'],
            'ARID1A':['TP53','CTCF','PTEN']
        },
         'luad':{'KRAS':['STK11','NF1'],
                'TP53':['STK11'],
        },
         },
    'abctb':{
        'brca':{
            'ER':['PR'],
            'PR':['ER']
        }
    }
}

