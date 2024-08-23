
import argparse
import os
import sys
import pathlib
import torch
from pathlib import Path
import time
import pandas as pd
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Passing tissue name and'
            ,'worksapce dir and output tissue segment dir'))
    parser.add_argument('--gpu', type=str, nargs='?', const='0', default='0')
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
    parser.add_argument('--out_temp', type=str,
                         default=f'{OUTPUT_DIR}/TEMP')
    parser.add_argument('--exp_tag', type=str,
                         default='tissue_segment_')
    parser.add_argument('--batch_size', type=int,
                         default=16)
    parser.add_argument('--num_workers', type=int,
                         default=32)
    parser.add_argument('--wsi_bulk_idx', type=int, default=0)
    parser.add_argument('--wsi_num_bulk', type=int, default=0)

    args = parser.parse_args()
    print('Printing args dictionary',args)
    WORKSPACE_DIR = args.workspace_dir
    WSI_DIR = args.wsi_dir  
    META_FILE = args.wsi_meta
    GDC_MANIFEST = args.gdc_manifest
    RAW_DIR = args.raw_dir 
     
    #import tiatoolbox
    from skimage.io import imread
    from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
    from tiatoolbox.wsicore.wsireader import WSIReader,WSIMeta,VirtualWSIReader
    from tiatoolbox.models import WSIStreamDataset
    from net_desc import FCN_Model
    import numpy as np
    from cutils import (mkdir,rmdir,
                        convert_pytorch_checkpoint, difference_filename,
                            rm_n_mkdir, recur_find_ext,tqdm,shutil)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    PRETRAINED = (
        f"{WORKSPACE_DIR}/extract/pretrained/net_epoch-000005.tar"
    )

    pretrained = torch.load(PRETRAINED)['desc']
    pretrained = convert_pytorch_checkpoint(pretrained)

    model = FCN_Model()
    model.load_state_dict(pretrained)

    class WSIStreamDatsetX(WSIStreamDataset):
           def _get_reader(self, img_path):
                """Get appropriate reader for input path."""
                img_path = pathlib.Path(img_path)
                if self.mode == "wsi":
                    reader= WSIReader.open(img_path)
                    # Setting the mmp value manually
                    if not reader.info.mpp:
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
                    return reader
                img = imread(img_path)
                # initialise metadata for VirtualWSIReader.
                # here, we simulate a whole-slide image, but with a single level.
                metadata = WSIMeta(
                    mpp=np.array([1.0, 1.0]),
                    objective_power=10,
                    axes="YXS",
                    slide_dimensions=np.array(img.shape[:2][::-1]),
                    level_downsamples=[1.0],
                    level_dimensions=[np.array(img.shape[:2][::-1])],
                )
                return VirtualWSIReader(
                    img,
                    info=metadata,
                )

    class SemanticSegmentorX(SemanticSegmentor):

        @staticmethod
        def get_reader(img_path: str, mask_path: str, mode: str, auto_get_mask: bool):
            """Define how to get reader for mask and source image."""
            if not isinstance(img_path, np.ndarray):
                img_path = pathlib.Path(img_path)
            reader = WSIReader.open(img_path)

            # Setting the mmp value manually
            if not reader.info.mpp:
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

            mask_reader = None
            if isinstance(mask_path, WSIReader):
                mask_reader = mask_path
            elif mask_path is not None:
                if os.path.isfile(mask_path):
                    mask = imread(mask_path)  # assume to be gray
                else:
                    mask = mask_path
                mask = np.array(mask > 0, dtype=np.uint8)

                mask_reader = VirtualWSIReader(mask)
                mask_reader.info = reader.info
            elif auto_get_mask and mode == "wsi" and mask_path is None:
                # if no mask provided and `wsi` mode, generate basic tissue
                # mask on the fly
                mask_reader = reader.tissue_mask(
                    "morphological", resolution=1.25, units="power"
                )
                mask_reader.info = reader.info
            return reader, mask_reader

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {'units': 'mpp', 'resolution': 8.0},
        ],
        output_resolutions=[
            {'units': 'mpp', 'resolution': 8.0},
        ],
        save_resolution={'units': 'mpp', 'resolution': 8.0},
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[256, 256],
    )

    segmentor = SemanticSegmentorX(
        model=model,
        num_loader_workers=args.num_workers,
        batch_size=args.batch_size,
        dataset_class=WSIStreamDatsetX
    )

    # ! you can manually provide path here
    wsi_list = recur_find_ext(WSI_DIR, ['.svs', '.ndpi', '.tif'])
    processed_list = recur_find_ext(RAW_DIR, ['.npy'])
    unprocessed_wsi_list = {pathlib.Path(v).stem for v in wsi_list}-{pathlib.Path(v).stem for v in processed_list}

    # Reading wsi meta data file
    metaDf = pd.read_excel(META_FILE,index_col='wsi_name')

    #Reading GDC manifest
    manifestDf = pd.read_csv(GDC_MANIFEST,delimiter='\t')
    manifestDf.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    manifestDf['wsi_name']=[Path(p).stem for p in manifestDf['File Name']]
    manifestDf.set_index('wsi_name',inplace=True)
    metaManifest = manifestDf.join(metaDf)

    #Updateding wsi_list based on unprocessed files
    wsi_list = {wsi_path  for wsi_path in wsi_list
                 if Path(wsi_path).stem in unprocessed_wsi_list and Path(wsi_path).stem in metaManifest.index.tolist()}

    print(f'Number of wsi to be processed {len(wsi_list)}')
    # Creating output mask directory if not already there
    mkdir(RAW_DIR)
      
    # Segmenting WSI-tissue region
    for wsi_path in tqdm(wsi_list):
        wsi_name = Path(wsi_path).stem

        NETWORK_DATA = False
        CACHE_DIR = args.out_temp
        cache_wsi_dir = f"{CACHE_DIR}/wsi-cache"
        cache_out_dir = f"{CACHE_DIR}/TEMP{args.exp_tag}"
        mkdir(cache_wsi_dir)

        if NETWORK_DATA:
            stime = time.perf_counter()
            cache_wsi_path = f"{cache_wsi_dir}/{wsi_name}"
            shutil.copyfile(wsi_path, cache_wsi_path)
            etime = time.perf_counter()
            print(f"Copying to local storage: {etime - stime}")
        else:
            cache_wsi_path = wsi_path
        rmdir(cache_out_dir)
        
        wsi_segment_mapping = segmentor.predict(
            [cache_wsi_path],
            masks=None,
            mode='wsi',
            on_gpu=True,
            ioconfig=ioconfig,
            crash_on_exception=True,
            save_dir=cache_out_dir
                        )
        if wsi_segment_mapping:
            wsi_segment_mapping = wsi_segment_mapping[0]
            out_npy,_ = wsi_segment_mapping
            shutil.move(f'{wsi_segment_mapping[1]}.raw.0.npy',
                        f'{RAW_DIR}/{wsi_name}.npy')
        rmdir(cache_wsi_path)

 
