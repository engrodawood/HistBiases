import numpy as np
import inspect
import logging
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib
import re
from tqdm import tqdm
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader,OpenSlideWSIReader
from tiatoolbox.tools.patchextraction import PatchExtractor
from skimage.io import imread, imsave
from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor,DeepFeatureExtractor
import matplotlib.pyplot as plt
from tiatoolbox.tools import patchextraction
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
import torch
from termcolor import colored
# from torch_geometric.data import Data
from torch.autograd import Variable
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
import pickle
import pandas as pd
import argparse


def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """
    recursively find all files in directories end with the `ext`
    such as `ext='.png'`

    return list is alrd sorted
    """
    assert isinstance(ext_list, list)
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def log_debug(msg):
    (
        frame, filename, line_number, function_name, lines, index
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]

    indentation_level = line.find(line.lstrip())
    logging.debug('{i} {m}'.format(
            i='.'*indentation_level,
            m=msg
        ))


def log_info(msg):
    (
        frame, filename, line_number, function_name, lines, index
    ) = inspect.getouterframes(inspect.currentframe())[1]

    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info('{i} {m}'.format(
            i='.'*indentation_level,
            m=msg
        ))


def wrap_func(idx, func, *args):
    try:
        return idx, func(*args)
    except Exception as e:    
        return e, idx, None


def multiproc_dispatcher(
        data_list,
        num_workers=0,
        show_pbar=True, 
        crash_on_exception=False):
    """
    data_list is alist of [[func, arg1, arg2, etc.]]
    Resutls are alway sorted according to source position
    """
    if num_workers > 0:
        proc_pool = ProcessPoolExecutor(num_workers)

    result_list = []
    future_list = []

    if show_pbar:
        pbar = tqdm(total=len(data_list), ascii=True, position=0)
    for run_idx, dat in enumerate(data_list):
        func = dat[0]
        args = dat[1:]
        if num_workers > 0:
            future = proc_pool.submit(
                wrap_func, run_idx, func, *args)
            future_list.append(future)
        else:
            # ! assume 1st return is alwasy run_id
            result = wrap_func(run_idx, func, *args)
            if len(result) == 3 and crash_on_exception:
                raise result[0]
            elif len(result) == 3:
                result = result[1:]
            result_list.append(result)
            if show_pbar:
                pbar.update()
    if num_workers > 0:
        for future in as_completed(future_list):
            if future.exception() is not None:
                if crash_on_exception:
                    raise future.exception()
                logging.info(future.exception())
            else:
                result = future.result()
                if len(result) == 3 and crash_on_exception:
                    raise result[0]
                elif len(result) == 3:
                    result = result[1:]
                result_list.append(result)
            if show_pbar:
                pbar.update()
        proc_pool.shutdown()
    if show_pbar:
        pbar.close()

    result_list = sorted(result_list, key=lambda k : k[0])
    result_list = [v[1:] for v in result_list]
    return result_list


def intersection_filename(listA, listB):
    """Return paths with file name exist in both A and B."""
    name_listA = [pathlib.Path(v).stem for v in listA]
    name_listB = [pathlib.Path(v).stem for v in listB]
    union_name_list = list(set(name_listA).intersection(set(name_listB)))
    union_name_list.sort()

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listA.index(name))
        except ValueError:
            pass
    if len(sel_idx_list) == 0:
        return [], []
    sublistA = np.array(listA)[np.array(sel_idx_list)]

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listB.index(name))
        except ValueError:
            pass
    sublistB = np.array(listB)[np.array(sel_idx_list)]

    return sublistA.tolist(), sublistB.tolist()

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(
        v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

def difference_filename(listA, listB):
    """Return paths in A that dont have filename in B."""
    name_listB = [pathlib.Path(v).stem for v in listB]
    name_listB = list(set(name_listB))
    name_listA = [pathlib.Path(v).stem for v in listA]
    sel_idx_list = []
    for idx, name in enumerate(name_listA):
        try:
            name_listB.index(name)
        except ValueError:
            sel_idx_list.append(idx)
    if len(sel_idx_list) == 0:
        return []
    sublistA = np.array(listA)[np.array(sel_idx_list)]
    return sublistA.tolist()

def toTensor(v,dtype = torch.float,
            requires_grad = True,
            device='cpu'):  
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)

def toGeometric(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def writePickle(ofile,G):
    with open(ofile, 'wb') as f:
          pickle.dump(G, f)
