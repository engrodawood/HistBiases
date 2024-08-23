import argparse
import os
import sys
import pathlib
import torch
from pathlib import Path
import time
from net_desc import CustomCNN,RetCCL
import numpy as np
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor,DeepFeatureExtractor
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader,OpenSlideWSIReader
from net_desc import FCN_Model,CTransPath
from tiatoolbox.tools import patchextraction
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imsave
from cutils import (mkdir,rmdir,
                        difference_filename,
                        rm_n_mkdir, recur_find_ext,tqdm,shutil)
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR

def select_patch(reader: VirtualWSIReader, bounds,
                 resolution, units):
    """Accept coord as long as its box contains bits of mask."""
    selected = []
    for bound in bounds:
        roi = reader.read_bounds(
            bound,
            resolution=resolution,
            units=units,
            interpolation="nearest",
            coord_space="resolution",
        )
        selected.append(np.mean(roi > 0) > 0.4)
    # import pdb; pdb.set_trace()
    return np.array(selected).astype('bool')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Passing tissue name and'
            ,'worksapce dir and also graphs dir'))
    parser.add_argument('--gpu', type=str, nargs='?', const='0', default='0')
    parser.add_argument('--workspace_dir', type=str,
                         default=WORKSPACE_DIR)
    parser.add_argument('--wsi_dir', type=str,
                         default=f'{DATA_DIR}/WSIs')
    parser.add_argument('--wsi_meta', type=str,
                         default=f'{DATA_DIR}/wsi_meta_mfilled.xlsx')
    parser.add_argument('--gdc_manifest', type=str,
                         default=f'{DATA_DIR}/gdc_sample_sheet_mutation.tsv')
    parser.add_argument('--mask_dir', type=str,
                         default=f'{OUTPUT_DIR}/TISSUE_MASKS')
    parser.add_argument('--out_temp', type=str,
                         default=f'{OUTPUT_DIR}/TEMP')
    parser.add_argument('--feat_type', type=str,
                         default='DF')
    parser.add_argument('--out_feats', type=str,
                         default=f'{OUTPUT_DIR}/FEATS/')# Change this Path
    parser.add_argument('--batch_size', type=int,
                         default=32)
    parser.add_argument('--num_workers', type=int,
                         default=16)
    parser.add_argument('--exp_tag', type=str,
                         default='Job0')
    
    args = parser.parse_args()
    # Configuration for ShuffleNet/Deep Features

    print('Printing args dictionary',args)
    WORKSPACE_DIR = args.workspace_dir
    WSI_PATH = args.wsi_dir
    TISSUE_MASK_PATH = args.mask_dir
    META_FILE = args.wsi_meta
    GDC_MANIFEST = args.gdc_manifest
    FEAT_TYPE = args.feat_type
    OUT_DIR = f'{args.out_feats}/{FEAT_TYPE}'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    if FEAT_TYPE in ['DF']:
        config = {'wsi-format': '.svs', 'patch-size': [512, 512], 'stride-shape': [512, 512],
                'read-resolution': 0.5, "units": 'mpp', 
                'num_wokers':args.num_workers,'batch_size':args.batch_size
                }
        encoder = shuffleNet = CustomCNN()
        
    else:
        config = {'patch-size': [1024, 1024], 'stride-shape': [1024, 1024],
                    'read-resolution': 0.5, "units": 'mpp',
                    'num_wokers':args.num_workers,'batch_size':args.batch_size
                    }
        encoder = CTransPath(
            checkpoint_path=f'{WORKSPACE_DIR}/extract/pretrained/ctranspath.pth'
              )

     # Reading wsi meta data file
    metaDf = pd.read_excel(META_FILE,index_col='wsi_name')

    #Reading GDC manifest
    manifestDf = pd.read_csv(GDC_MANIFEST,delimiter='\t')
    manifestDf.replace({'TCGA-READ':'TCGA-COLON','TCGA-COAD':'TCGA-COLON'},inplace=True)
    manifestDf['wsi_name']=[Path(p).stem for p in manifestDf['File Name']]
    manifestDf.set_index('wsi_name',inplace=True)
    metaManifest = manifestDf.join(metaDf)
    
    mkdir(OUT_DIR)
    
   
    from skimage.io import imread
    from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor,DeepFeatureExtractor
    from tiatoolbox.wsicore.wsireader import WSIReader,WSIMeta,VirtualWSIReader
    from tiatoolbox.models import WSIStreamDataset
    
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

    class DeepFeatureExtractorX(DeepFeatureExtractor):

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


    extractor = DeepFeatureExtractorX(
        model = encoder,# Encoder of your choice
        batch_size=config['batch_size'],
        num_loader_workers=config['num_wokers'],
        dataset_class=WSIStreamDatsetX
    )
    

    wsi_names = recur_find_ext(f'{WSI_PATH}/', [config['wsi-format']])
    wsi_names = [pathlib.Path(v).stem for v in wsi_names]

    output_mapping = []
    runs = 0
    skipList = []
    for wsi_name in tqdm(wsi_names):
        wsi_path = f'{WSI_PATH}/{wsi_name}{config["wsi-format"]}'
        mask_path = f"{TISSUE_MASK_PATH}/{wsi_name}.png"
        
        # Checking if mask is available
        if not os.path.isfile(mask_path):
            print('Tissue mask is not processed yes')
            continue
        
        # Checking if the wsi is already processed
        if os.path.isfile(f'{OUT_DIR}/{wsi_name}_feat.npy'):
            continue

        print('WSI Processing', wsi_name)

        ioconfig = IOSegmentorConfig(
            input_resolutions=[
                {"units": config["units"],
                    "resolution": config["read-resolution"]},
            ],
            output_resolutions=[
                {"units": config["units"],
                    "resolution": config["read-resolution"]},
            ],
            patch_input_shape=config["patch-size"],
            patch_output_shape=config["patch-size"],
            stride_shape=config["stride-shape"],
        )
        extractor.filter_coordinates = select_patch
        
        NETWORK_DATA = False
        CACHE_DIR = args.out_temp
        cache_wsi_dir = f"{CACHE_DIR}/tcga"
        cache_out_dir = f"{CACHE_DIR}/TEMP_{args.exp_tag}"
        
        if NETWORK_DATA:
            stime = time.perf_counter()
            mkdir(cache_wsi_dir)
            cache_wsi_path = f"{cache_wsi_dir}/{wsi_name}{config['wsi-format']}"
            # Checking if file is not in the cache 
            # then copy it to the cache
            if not os.path.isfile(cache_wsi_path):
                print('Copying WSI to the cache')
                shutil.copyfile(wsi_path, cache_wsi_path)
                etime = time.perf_counter()
                print(f"Copying to local storage: {etime - stime}")
        else:
            cache_wsi_path = wsi_path
        rmdir(cache_out_dir)
        try:
            wsi_feat_pair = extractor.predict(
                [cache_wsi_path],
                [mask_path],
                mode="wsi",
                on_gpu=True,
                ioconfig=ioconfig,
                crash_on_exception=True,
                save_dir=f"{cache_out_dir}",
            )
            # append the mapping to the list
            if wsi_feat_pair:
                output_mapping.extend(wsi_feat_pair)
                wsi_feat_pair = wsi_feat_pair[0]
                out_filename = Path(wsi_feat_pair[0]).stem
                mkdir(OUT_DIR)
                shutil.copyfile(
                    f'{wsi_feat_pair[1]}.features.0.npy', f'{OUT_DIR}/{out_filename}_feat.npy')
                shutil.copyfile(
                    f'{wsi_feat_pair[1]}.position.npy', f'{OUT_DIR}/{out_filename}_pos.npy')
                rmdir(cache_out_dir)
                runs += 1
        except:
            skipList.append(wsi_name)
            pd.DataFrame(skipList).to_csv(f'{OUT_DIR}/wsi_failed_features.csv')
            rmdir(cache_out_dir)
        # deleting cache directory
        rmdir(cache_wsi_dir)