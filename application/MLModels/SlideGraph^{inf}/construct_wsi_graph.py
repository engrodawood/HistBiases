#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:59:38 2021

@author: dawood
"""

# from cutils import *
import torch; print(torch.__version__)
import torch; print(torch.version.cuda)
import numpy as np
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
from sklearn.neighbors import KDTree as sKDTree
from tqdm import tqdm
from pathlib import Path
import pandas as pd 
import argparse
from gutils import *
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR


def connectPatches(C,dthresh = 3000):
    '''
    Taken from original SlideGraph+ repo
    '''
    tess = Delaunay(C)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx = neighbors    
    W = np.zeros((C.shape[0],C.shape[0]))
    for n in nx:
        nx[n] = np.array(list(nx[n]),dtype = np.int32)
        nx[n] = nx[n][KDTree(C[nx[n],:]).query_ball_point(C[n],r = dthresh)]
        W[n,nx[n]] = 1.0
        W[nx[n],n] = 1.0        
    return W 

# Will work on LSF workspace      

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dthreshold', type=int,  default=4000,
                        help='Distance thresold for connecting patches')
    parser.add_argument('--feats_dir', type=str,
                         default=f'{OUTPUT_DIR}/FEATS/DF',# By defulat using Deep Features
                         help=('input directory where position and feature of wsi tiles are strored',
                         'wsiname_feat.npy patch level representation',
                         'wsiname_pos.npy: rectangular bounding box from where patch was extracted.'))
    parser.add_argument('--out_dir', type=str,
                         default=f'{OUTPUT_DIR}/GraphsDF',
                         help='Path where to the generated graphs')

    args = parser.parse_args()

    FEAT_POS_DIR = args.feats_dir
    dthreshold = args.dthreshold
    OUTPUT_DIR = f'{args.out_dir}/'

    mkdir(OUTPUT_DIR)

    # Getting list of wsi_names
    wsi_list = list(
        {
        Path(f).stem.replace('_pos','').replace('_feat','') for f in recur_find_ext(FEAT_POS_DIR, ['.npy'])
        }
    )
    #Debug
    # wsi_list = wsi_list[0:100]
    rStats = pd.DataFrame(
        np.zeros(((len(wsi_list),2))),
                 columns=['wsi-name','no-patches']
                 )
    rStats['wsi-name'] = wsi_list
    rStats.set_index('wsi-name',inplace=True)
    # Interacting though each file and generate the graph
    for wsi_name in tqdm(wsi_list):
        print(f'~~~~~~~~~~~~~ Processing ~~~~~~~~ {wsi_name}')

        coord_file = f'{FEAT_POS_DIR}/{wsi_name}_pos.npy'
        feat_file = f'{FEAT_POS_DIR}/{wsi_name}_feat.npy'
        ofile = f'{OUTPUT_DIR}/{wsi_name}.pkl'

        if os.path.isfile(ofile):
            print('file ID conflict')
            continue
            import pdb; pdb.set_trace()
        try:
            F = np.load(feat_file)
            Bbox = np.load(coord_file)

            rStats.loc[wsi_name,'no-patches'] = Bbox.shape[0]
            # import pdb; pdb.set_trace()
            # Getting connectivity matrix
            W = connectPatches(Bbox[:,:2],# Using top left corner
                            dthresh=dthreshold)
            # Contructing graph
            G = toGeometric(F,W,y=1)
            G.coords = toTensor(Bbox[:,:2],requires_grad=False)
            writePickle(ofile,G)
            rStats.to_csv(f'{OUTPUT_DIR}/patch_stats.csv')
        except:
            print('Graph Not generated')
    print()
    print()
