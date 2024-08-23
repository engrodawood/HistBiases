
import shutil
import pathlib
import os
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
import pickle

def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return

def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return

def toTensor(v,dtype = torch.float,requires_grad = True): 
    device = 'cpu'   
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)

def toGeometric(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))


def recur_find_ext(root_dir, ext_list,rev=False):
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
    file_path_list.sort(reverse=rev)
    return file_path_list


def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def writePickle(ofile,G):
    with open(ofile, 'wb') as f:
          pickle.dump(G, f)
