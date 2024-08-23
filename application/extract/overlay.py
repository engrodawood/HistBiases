# %%
import argparse
import pathlib
import sys

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time
import os
from skimage.morphology import binary_closing, binary_erosion, binary_opening

from cutils import (mkdir,rmdir,tqdm,shutil,
                    convert_pytorch_checkpoint, difference_filename,
                        rm_n_mkdir, recur_find_ext,multiproc_dispatcher)

import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader,WSIMeta

mpl.rcParams['figure.dpi']= 300


def find_file_in_dir(root_dir, base_name, ext):
    paths = recur_find_ext(root_dir, [ext])
    base_names = [pathlib.Path(v).stem for v in paths]
    try:
        index = base_names.index(base_name)
        result_path = paths[index]
    except ValueError:
        result_path = None
    return result_path

def process_one(idx,wsi_name):
    wsi_path = find_file_in_dir(WSI_DIR, wsi_name, '.svs')
    ANALYSIS=False
    if CACHE_DIR:
        # Downloding WSI to cache as the file is on network
        stime = time.perf_counter()
        cache_wsi_dir = f'{CACHE_DIR}/{idx}'
        mkdir(cache_wsi_dir)
        cache_wsi_path = f"{cache_wsi_dir}/{wsi_name}.svs"

        # Checking if file is not in the cache 
        # then copy it to the cache
        if not os.path.isfile(cache_wsi_path):
            print('Copying WSI to the cache')
            # import pdb; pdb.set_trace()
            shutil.copyfile(wsi_path, cache_wsi_path)
            etime = time.perf_counter()
            print(f"Copying to local storage: {etime - stime}")
    else:
        cache_wsi_path = wsi_path
    # import pdb; pdb.set_trace()
    reader = OpenSlideWSIReader(cache_wsi_path)

    # Setting the mmp value manually
    if not np.any(reader.info.mpp):
        '''
        Setting the mmp manually
        '''
        old_info = reader.info

        m_mpp = metaManifest.loc[wsi_name,'x_mpp']
        m_objective = metaManifest.loc[wsi_name,'objective_power']

        new_info = WSIMeta(
            axes=old_info.axes,
            slide_dimensions=old_info.slide_dimensions,
            level_dimensions=old_info.level_dimensions,
            objective_power=m_objective,
            mpp = np.array([m_mpp,m_mpp])
                        )
        reader.info=new_info
    raw = np.load(f'{RAW_DIR}/{wsi_name}.npy')
    raw = VirtualWSIReader(raw, mode='bool')
    raw.info = reader.info
    raw_thumb = raw.slide_thumbnail(**PLOT_RESOLUTION)
    wsi_thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
    proc = raw_thumb[..., 1] > 0.5
    proc_rgb = np.zeros_like(wsi_thumb)
    proc_rgb[proc > 0] = 255
    cv2.imwrite(f'{OUT_MASK_DIR}/{wsi_name}.png', cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR))
    if ANALYSIS:
        overlaid = alpha * proc_rgb + (1 - alpha) * wsi_thumb
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        cv2.imwrite(f'{OUT_MASK_DIR}/{wsi_name}.thumb.png', cv2.cvtColor(wsi_thumb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{OUT_MASK_DIR}/{wsi_name}.overlaid.png', cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    # Deleting the WSI after processed
    if CACHE_DIR:
        rmdir(cache_wsi_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Passing tissue name and'
        ,'worksapce dir and also graphs dir'))
    parser.add_argument('--workspace_dir', type=str,
                         default=WORKSPACE_DIR)
    parser.add_argument('--wsi_dir', type=str,
                         default=f'{DATA_DIR}/WSIs')
    parser.add_argument('--wsi_meta', type=str,
                         default=f'{DATA_DIR}/META_WSIs/wsi_meta_mfilled.xlsx')
    parser.add_argument('--gdc_manifest', type=str,
                         default=f'{DATA_DIR}/META_WSIs/gdc_sample_sheet_mutation.tsv')
    parser.add_argument('--raw_dir', type=str,
                         default=f'{OUTPUT_DIR}/raw')
    parser.add_argument('--out_mask', type=str,
                         default=f'{OUTPUT_DIR}/TISSUE_MASKS')
    parser.add_argument('--out_temp', type=str,
                         default=None)
    
    args = parser.parse_args()
    print('Printing args dictionary',args)
    WORKSPACE_DIR = args.workspace_dir
    WSI_DIR = args.wsi_dir  
    META_FILE = args.wsi_meta
    GDC_MANIFEST = args.gdc_manifest
    OUT_MASK_DIR = args.out_mask
    RAW_DIR = args.raw_dir
    
    wsi_list = recur_find_ext(WSI_DIR, ['.svs', '.ndpi', '.tif'])
    # Deleting temp directory and making out_seg directory
    mkdir(OUT_MASK_DIR)
    
    NETWORK_DATA = False
    CACHE_DIR = args.out_temp    

    names = recur_find_ext(RAW_DIR, ['.npy'])
    names = set([pathlib.Path(v).stem for v in wsi_list]).intersection([pathlib.Path(v).stem for v in names])
    processed_list = recur_find_ext(OUT_MASK_DIR, ['.png'])
    processed_list = {pathlib.Path(v).stem for v in processed_list}
    names = list(names-processed_list)

    # Reading wsi meta data file
    import pandas as pd
    from pathlib import Path
    metaDf = pd.read_excel(META_FILE,index_col='wsi_name')

    #Reading GDC manifest
    manifestDf = pd.read_csv(GDC_MANIFEST,delimiter='\t')
    manifestDf.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    manifestDf['wsi_name']=[Path(p).stem for p in manifestDf['File Name']]
    manifestDf.set_index('wsi_name',inplace=True)
    metaManifest = manifestDf.join(metaDf)

    alpha = 0.4
    cmap = plt.get_cmap('jet')
    PLOT_RESOLUTION = dict(resolution=8.0, units='mpp')

    runs = []
    for idx,base_name in enumerate(names):
        print(base_name)
        runs.append([process_one, idx,base_name])
    multiproc_dispatcher(runs, num_workers=0, crash_on_exception=True)
