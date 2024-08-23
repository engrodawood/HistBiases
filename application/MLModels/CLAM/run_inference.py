import argparse, os



for voi in range(200):
        EXP_CODE = f'G{voi}'

        DATA_DIR = '/data/PanCancer/HTEX_repo/CLAM/data_abctb'
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        pwd = os.path.dirname(os.path.realpath(__file__))
        # Experiment arguments
        parser.add_argument('--pwd', default=pwd, help='path of the current folder.')
        parser.add_argument('--exp_code', default=EXP_CODE, help='Experiment code.')
        parser.add_argument('--split_dir', default=f'{DATA_DIR}/G{voi}/n_splits/', help='path to the file which contains the split of train and val.')
        parser.add_argument('--label_path', default=f'{DATA_DIR}/G{voi}/labels/G{voi}_clean.csv', help='path to the file which contains the labels of the dataset.')
        parser.add_argument('--exp_name', default=f'{EXP_CODE}', help='The name for this experiment') 
        parser.add_argument('--feature_dir', default='/data/PanCancer/FEATS/ABCTB-FEATS-SHUFFLE-BAGS', help='The path to the folder which contains extracted features.')
        parser.add_argument('--finetune_feat', default=False, type=bool, help='True for finetuned features, False for ImageNet features.')
        parser.add_argument('--result_dir', default=f'{pwd}/results/', help='The result path.') 
        parser.add_argument('--output_dir', default=f'{pwd}/abctb_results/', help='The result path.') 
        parser.add_argument('--input_dim', default=1024, type=int) 
        parser.add_argument('--gpu_id', default='1', help='GPU id.') 

        parser.add_argument('--lr', default=2e-4, type=float) 
        parser.add_argument('--fold_num', default=5, type=int) 
        parser.add_argument('--task', default='task_1_tumor_vs_normal', type=str) 
        parser.add_argument('--model_type', default='clam_sb', type=str) 

        args = parser.parse_args()

        command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} python {pwd}/eval.py " \
                f"--drop_out --k {args.fold_num}  --models_exp_code {args.exp_code}_s1 " \
                f"--save_exp_code {args.exp_code}_s1 --task {args.task} --model_type {args.model_type} " \
                f"--results_dir {args.result_dir} --output_dir {args.output_dir} --data_root_dir {args.feature_dir} " \
                f"--splits_dir {args.split_dir} --csv_path {args.label_path} --finetune_feat {args.finetune_feat} " \
                f"--embedding_size {args.input_dim} "

        os.system(command)