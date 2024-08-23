#! /usr/bin/env python

#
# Copyright (c) 2018 Memorial Sloan Kettering Cancer Center.
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
# documentation provided hereunder is on an "as is" basis, and
# Memorial Sloan Kettering Cancer Center
# has no obligations to provide maintenance, support,
# updates, enhancements or modifications.  In no event shall
# Memorial Sloan Kettering Cancer Center
# be liable to any party for direct, indirect, special,
# incidental or consequential damages, including lost profits, arising
# out of the use of this software and its documentation, even if
# Memorial Sloan Kettering Cancer Center
# has been advised of the possibility of such damage.
#
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
import sys
import os
import argparse
import math
import numpy
import pandas as pd

LOG_STR = ""

# WGS/WES CDS
WES_WGS_CDS = 30

# egilible variants (for counting mutations)
VC_NONSYNONYMOUS_LIST = [
	"Frame_Shift_Del", 
	"Frame_Shift_Ins", 
	"In_Frame_Del", 
	"In_Frame_Ins", 
	"Missense_Mutation", 
	"Nonsense_Mutation", 
	"Splice_Site", 
	"Nonstop_Mutation", 
	"Splice_Region"
]

# names - case list file
SEQ_CASE_LIST_FILE_NAME = "cases_sequenced.txt"
SEQ_CASE_LIST_FIELD_NAME = "case_list_ids:"

# names - MAF file
MAF_FILE_NAME = "data_mutations.txt"
MAF_VC_COL_ID = "Variant_Classification"
MAF_ID_COL_ID = "Tumor_Sample_Barcode"
MAF_SOMATIC_COL_ID = "Mutation_Status"

# names  - matrix file
MATRIX_FILE_NAME = "data_gene_panel_matrix.txt"
MATRIX_SAMPLE_ID_COL = "SAMPLE_ID"
MATRIX_MUT_COL = "mutations"

# names - clinical file
CLIN_INPUT_FILE_NAME = "data_clinical_sample.txt"

CLIN_ID_COL_ID = "SAMPLE_ID"
CLIN_TMB_COL_ID = "TMB_NONSYNONYMOUS"

###### 
# count eligible variants for all samples from MAF
######
def getTMB(_inputStudyPath,
				selectedGenes = []# By default no gene is ignored. 
				):

	_result = {}
	with open(_inputStudyPath + "/" + MAF_FILE_NAME,'r') as _maf:
		_headers = []
		_posSampleID = -1
		_posVariantClass = -1

		for _line in _maf:
			if _line.startswith('#'):
				continue
			elif _line.startswith('Hugo_Symbol'):
				# parsing MAF header
				_headers = _line.rstrip("\n").rstrip('\r').split('\t')
				_posSampleID = _headers.index(MAF_ID_COL_ID)
				_posVariantClass = _headers.index(MAF_VC_COL_ID)
				_selectedGene = _headers.index('Hugo_Symbol')
			else:	
				# parsing content
				_items = _line.rstrip("\n").rstrip('\r').split('\t')

				_sampleID = _items[_posSampleID]
				_variantClass = _items[_posVariantClass]
				_mutGene = _items[_selectedGene]
				if _mutGene not in selectedGenes:continue

				if _variantClass in VC_NONSYNONYMOUS_LIST :
					if _sampleID in _result:
						_result[_sampleID]["vc_count"] = _result[_sampleID]["vc_count"] + 1
					else:
						_result[_sampleID] = {
							'vc_count': 1,
						}
						
	# Getting IDS of samples that are sequenced
	return _result
	



data = []
import numpy as np
for idx,study in enumerate(['brca','coadread','luad','ucec']):
    _inputStudyFolder = f'/data/PanCancer/GeneGrouping/MolecularData/PanCancerAtlas/unzipped/{study}_tcga_pan_can_atlas_2018'#args.input_study_folder
    sys.stdout.write(os.path.basename(_inputStudyFolder) + "\t")
    # Reading frequently oncogenes in cancer
    Oc = pd.read_csv('/data/PanCancer/HistBiases/data/MUT/Oncogenes.csv',delimiter='\t')
    genes = Oc[Oc.iloc[:,-1]=='Yes']['Gene'].tolist()
    resdict = getTMB(_inputStudyFolder,genes)
    import matplotlib.pyplot as plt
    CC = pd.DataFrame.from_dict(resdict).T.to_numpy().ravel()
    print(CC.mean(),np.median(CC))
    low = np.percentile(CC,25)
    high = np.percentile(CC,75)
    print(CC.shape)
    CC = CC[(CC>low)*(CC<high)]
    print(CC.shape)
    plt.violinplot(CC,[idx],showmeans=True,showextrema=True,points=500)
plt.savefig('Counts.png')
print()