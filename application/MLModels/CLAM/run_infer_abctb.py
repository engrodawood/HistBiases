import os
CUDA = '0'
import sys
sys.path.append('.')
from application.project import CLAM_DICT_EXTRV_ABCTB,TISSUE_TYPES_HIST
from application.utils.utilIO import mkdir
print(CLAM_DICT_EXTRV_ABCTB)
#Repr = 'CTransPath'
Repr = 'SHUFFLE'
for tissue in ['brca']:
    DATA_DIR = f'{CLAM_DICT_EXTRV_ABCTB[Repr]["OUT_SPLITS_DIR"]}/{tissue}/'
    FEATURES_DIR = f'{CLAM_DICT_EXTRV_ABCTB[Repr]["OUT_BAGS_DIR"]}/'
    OUT_PATH = f'{CLAM_DICT_EXTRV_ABCTB[Repr]["OUT_DIR"]}/{tissue}/'
    CHECKPOINTS_PATH = f'{CLAM_DICT_EXTRV_ABCTB[Repr]["CHECKPOINTS"]}/{tissue}/'
    mkdir(OUT_PATH)
    for gene in ['HER2']:#os.listdir(DATA_DIR):
        EXP_CODE = f'{str.upper(tissue)}_{gene}'
        SPLIT_DIR = f'{DATA_DIR}/{gene}/n_splits/'
        CSV_PATH = f'{DATA_DIR}/{gene}/labels/{gene}_clean.csv'  

        pwd = os.path.dirname(os.path.realpath(__file__))
        LR = 2e-4
        K = 4
        LABEL_FRAC = -1
        BAG_LOSS = 'ce'
        INST_LOSS = 'svm'
        TASK = 'task_1_tumor_vs_normal'
        EMBEDDING_SIZE = CLAM_DICT_EXTRV_ABCTB[Repr]["EMBEDDING_SIZE"]
        MODEL_TYPE = 'clam_sb'

        command = f"CUDA_VISIBLE_DEVICES={CUDA} python {pwd}/eval.py" \
                f" --drop_out --k {K} --models_exp_code {EXP_CODE}_s1" \
                f" --save_exp_code {EXP_CODE}_eval --task {TASK} --model_type {MODEL_TYPE}" \
                f" --results_dir {CHECKPOINTS_PATH} --output_dir {OUT_PATH}" \
                f" --data_root_dir {FEATURES_DIR} --splits_dir {SPLIT_DIR} --csv_path {CSV_PATH}" \
                f" --embedding_size {EMBEDDING_SIZE}" 
        os.system(command)