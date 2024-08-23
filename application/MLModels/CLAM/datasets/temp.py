import os

files = os.listdir('/data/remote_backup/DeepFeaturesBags/CTransPath/brca')

import numpy as np

for f in files:
    print(f)
    D = np.load(f'/data/remote_backup/DeepFeaturesBags/CTransPath/brca/{f}')
    print(D.shape)