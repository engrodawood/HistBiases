PROJECT_DIR  = '/data/PanCancer/HistBiases'
DATA_DIR = f'{PROJECT_DIR}/data'
WORKSPACE_DIR = f'{PROJECT_DIR}/application'
OUTPUT_DIR = f'{PROJECT_DIR}/output'

BIOMARKER_TYPE_COLOR_CONFIG = ['#009988','#EE7733','#EE3377','#997700']

CODEP_COLORS = {'low':"#4477AA",'medium':"#F7F7F7",'high':"#BB5566"}

CODEP_PLOT_DICT = {'TOPK':10,# How many genes to show
                   'SIZE':180,# Size of Scatter dot
                    'DONUT_SIZE':15,
                    'SIZE_OFFSET':300,
                   'COLOR':{'MUT':'#EE6677','WT':'#0077BB','BG':'#FFFFFF','H':'#BB5566','L':'#004488','M':'#006633',#F7F7F7',
                            'ALPHA':0.7,'BOX_FACE':'purple','BOX_TICK':'#FFFFFF'}, # Pie charts colors
                   'FONT_SIZE':12,
                    'FIG_SIZE':{'abctb_brca':(4,5),'tcga_brca':(8,5),'tcga_colon':(12,5),'cptac_brca':(8,4),'cptac_colon':(8,4),
                                'tcga_luad':(4,4),'tcga_ucec':(12,4),'cptac_luad':(4,4),'cptac_ucec':(12,4)},
                    'FIG_SIZE_TMB':{'abctb_brca':(5,6),'tcga_brca':(8,4),'tcga_colon':(5,3),'cptac_brca':(8,4),'cptac_colon':(5,3),
                                'tcga_luad':(7,4),'tcga_ucec':(5,3),'cptac_luad':(7,4),'cptac_ucec':(5,3)}
                   }

BAR_PLOT_DICT = {
                   'COLOR':['#BB5566','#004488','#0077BB','#EE6677'],
                   'COLOR_COMP':['#994455','#EE99AA','#004488','#6699CC'],
                   'FONT_SIZE':16,
                    'FIG_SIZE':{'tcga_brca':(6,5),'cptac_brca':(6,5),'abctb_brca':(3,5),'tcga_colon':(7,5),'cptac_colon':(5,5),
                                'tcga_ucec':(7,5),'cptac_ucec':(7,5)}
                   }

LAYER_DICT = {'SHUFFLE':'1024_1024_1024','CTransPath':'768_768_768'}

SGRAPH_INFER_DICT = {
    'CTransPath':{'TAG':'768_768_768','layers':[768,768,768],'bsize':8,
                                'target_dim':1,'folds':4,
                                'INFER_GRAPHS_CPTAC':'/data/CPTAC/Graphs/CPTAC_CTrans1024_05PP/dth_4000',
                                'INFER_GRAPHS_ABCTB':'/data/remote_backup/ABCTB/ABCTB_CTrans1024_05MPP_Graphs/dth_4000'
                                },
    'SHUFFLE':{'TAG':'1024_1024_1024','layers':[1024,1024,1024],'bsize':8,
                                'target_dim':1,'folds':4,
                                'INFER_GRAPHS_CPTAC':'/data/CPTAC/GRAPHS/20x-SHUFFLE_NET',
                                'INFER_GRAPHS_ABCTB':'/data/remote_backup/ABCTB/ABCTB_Shufl512_05MPP_Graphs/dth_4000'
                                }
}
                   
TISSUE_TYPES_HIST = ['luad','brca','colon','ucec']
TISSUE_TYPES_SEQ  =['coad','lusc','gbm','brca','paad','luad']

SORT_ORDER_ASSOC = {
            'brca':['ER','PR','TP53','CDH1','GATA3'],
            'colon':['MSI','BRAF','APC','RNF43','PTEN'],
            'luad':['TP53','STK11','EGFR','KRAS'],
            'ucec':['PTEN','TP53','CTNNB1','ARID1A','APC'],

              }

def get_experiments():
    models = ['SGraph','CLAM']
    representations = ['CTransPath','SHUFFLE']
    return {f'{r}_{m}': f'{r[0]}{m[0]}' if m == 'CLAM' else f'{r[0]}{m[1]}' for r in representations for m in models}


# CLAM DICTIONARY EXTERNAL VALIDATION
CLAM_DICT_EXTRV_CPTAC = {'CTransPath':
             {
                'IN_FEATS_DIR':'/data/CPTAC/FEATS/CPTAC_CTrans1024_05MPP',
                'OUT_BAGS_DIR':f'/data/CPTAC/FEATS/CPTAC_CTrans1024_05MPP_Bags',
                'OUT_SPLITS_DIR':f'{DATA_DIR}/CTransPath_CLAM_CPTAC/data',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/cptac/CLAM/CTransPath',
                'CHECKPOINTS':'/data/PanCancer/HistBiases/output/HistPred/tcga/CLAM/CTransPath/',
                'EMBEDDING_SIZE':768,
                 'folds':4
                 },
             'SHUFFLE':
             {
                'IN_FEATS_DIR':'/data/CPTAC/FEATS/20x-SHUFFLE_NET',
                'OUT_BAGS_DIR':f'/data/CPTAC/FEATS/20x-SHUFFLE_NET_Bags',
                'OUT_SPLITS_DIR':f'{DATA_DIR}/SHUFFLE_CLAM_CPTAC/data',
                'CHECKPOINTS':'/data/PanCancer/HistBiases/output/HistPred/tcga/CLAM/SHUFFLE/',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/cptac/CLAM/SHUFFLE',
                'EMBEDDING_SIZE':1024,
                'folds':4
                 }
             }

CLAM_DICT_EXTRV_ABCTB_DUP = {'CTransPath':
             {
                'IN_FEATS_DIR':'/data/remote_backup/ABCTB/ABCTB_CTrans1024_05MPP',
                'OUT_BAGS_DIR':f'/data/remote_backup/ABCTB/ABCTB_CTrans1024_05MPP_DUP',
                'OUT_SPLITS_DIR':f'{DATA_DIR}/CTransPath_CLAM_ABCTB_DUP/data',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/abctb_dup/CLAM/CTransPath/',
                'CHECKPOINTS':'/data/PanCancer/HistBiases/output/HistPred/tcga/CLAM/CTransPath/',
                'EMBEDDING_SIZE':768,
                 'folds':4
                 }
             }

CLAM_DICT_EXTRV_ABCTB = {'CTransPath':
             {
                'IN_FEATS_DIR':'/data/remote_backup/ABCTB/ABCTB_CTrans1024_05MPP',
                'OUT_BAGS_DIR':f'/data/remote_backup/ABCTB/ABCTB_CTrans1024_05MPP_Bags',
                'OUT_SPLITS_DIR':f'{DATA_DIR}/CTransPath_CLAM_ABCTB/data',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/abctb/CLAM/CTransPath/',
                'CHECKPOINTS':'/data/PanCancer/HistBiases/output/HistPred/tcga/CLAM/CTransPath/',
                'EMBEDDING_SIZE':768,
                 'folds':4
                 },
             'SHUFFLE':
             {
                'IN_FEATS_DIR':'/data/remote_backup/ABCTB/ABCTB_Shufl512_05MPP',
                'OUT_BAGS_DIR':f'/data/remote_backup/ABCTB/ABCTB_Shufl512_05MPP_Bags',
                'OUT_SPLITS_DIR':f'{DATA_DIR}/SHUFFLE_CLAM_ABCTB/data',
                'CHECKPOINTS':'/data/PanCancer/HistBiases/output/HistPred/tcga/CLAM/SHUFFLE/',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/abctb/CLAM/SHUFFLE',
                'EMBEDDING_SIZE':1024,
                'folds':4
                 }
             }

#CLAM OUTPUT BAG Features
CLAM_DICT= {'CTransPath':
             {
                'IN_FEATS_DIR':'/data/remote_backup/DeepFeatures/CTransPath',
                'OUT_BAGS_DIR':f'{DATA_DIR}/CTransPath_CLAM/Bags','OUT_SPLITS_DIR':f'{DATA_DIR}/CTransPath_CLAM/data',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/tcga/CTransPath_CLAM',
                'EMBEDDING_SIZE':768,
                 },
             'SHUFFLE':
             {
                'IN_FEATS_DIR':'/data/remote_backup/DeepFeatures/SHUFFLENET-FEATS-20x',
                'OUT_BAGS_DIR':f'{DATA_DIR}/SHUFFLE_CLAM/Bags','OUT_SPLITS_DIR':f'{DATA_DIR}/SHUFFLE_CLAM/data',
                'OUT_DIR':f'{OUTPUT_DIR}/HistPred/tcga/SHUFFLE_CLAM',
                'EMBEDDING_SIZE':1024
                 }
             }
