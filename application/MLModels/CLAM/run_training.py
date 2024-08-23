import os
CUDA = '1'
for voi in range(170,200,1):
    try:
        EXP_CODE = f'G{voi}'
        DATA_DIR = '/data/PanCancer/HTEX_repo/CLAM/data'
        FEATURES_DIR = f'/data/PanCancer/WSIs/brca/20x-FEAT-SHUFFLE-CUSTOM'
        SPLIT_DIR = f'{DATA_DIR}/G{voi}/n_splits/'
        CSV_PATH = f'{DATA_DIR}/G{voi}/labels/G{voi}_clean.csv'

        pwd = os.path.dirname(os.path.realpath(__file__))
        MAX_EPOCHS = 200
        LR = 2e-4
        K = 5
        LABEL_FRAC = -1
        BAG_LOSS = 'ce'
        INST_LOSS = 'svm'
        TASK = 'task_1_tumor_vs_normal'
        EMBEDDING_SIZE = 1024
        MODEL_TYPE = 'clam_sb'

        command = f"CUDA_VISIBLE_DEVICES={CUDA} python {pwd}/main.py --max_epochs {MAX_EPOCHS} --drop_out --early_stopping --lr {LR} --k {K} --label_frac {LABEL_FRAC} --weighted_sample --bag_loss {BAG_LOSS} --inst_loss {INST_LOSS} --task {TASK} --embedding_size {EMBEDDING_SIZE}  --exp_code {EXP_CODE} --model_type {MODEL_TYPE} --log_data --data_root_dir {FEATURES_DIR} --split_dir {SPLIT_DIR} --csv_path {CSV_PATH}"

        os.system(command)
    except:
        print(f'Group {voi} Inference failed')

