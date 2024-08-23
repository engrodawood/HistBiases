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
				ignoreGenes = [],# By default no gene is ignored.
				cohort = 'tcga',
				tissueType = 'brca'
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
				if _mutGene in ignoreGenes:continue

				if _variantClass in VC_NONSYNONYMOUS_LIST :
					if _sampleID in _result:
						_result[_sampleID]["vc_count"] = _result[_sampleID]["vc_count"] + 1
					else:
						_result[_sampleID] = {
							'vc_count': 1,
							'tmb': 0,
							'cds': WES_WGS_CDS # samples not in matrix are WGS/WES
						}
	# Coverting counts into TMB
	print()
	for _sample in _result.keys():
		_result[_sample]['tmb'] = _result[_sample]['vc_count']/_result[_sample]['cds']
	# Getting IDS of samples that are sequenced
	_caseListPath = _inputStudyPath + "/case_lists/" + SEQ_CASE_LIST_FILE_NAME
	with open(_caseListPath,'r') as _caseList:
		for _line in _caseList:
			if _line.startswith(SEQ_CASE_LIST_FIELD_NAME):
				_seqSampleIds = _line.split(':')[1].strip().split("\t")
	# Reading Clinical File
	_inputClincFilePath = _inputStudyPath + "/" + CLIN_INPUT_FILE_NAME
	clincDf = pd.read_csv(_inputClincFilePath,delimiter='\t',skiprows = 4, index_col='PATIENT_ID')
	clincDf = clincDf.loc[:,['TMB_NONSYNONYMOUS']]
	if cohort in ['tcga']:
		_seqSampleIds = [s[:12] for s in _seqSampleIds]
		clincDf.loc[_seqSampleIds,[f'TMB-{ignoreGenes[0]}']] = 0
		for pid in clincDf.index.tolist():
			if f'{pid}-01' in _result.keys():
				clincDf.loc[pid,[f'TMB-{ignoreGenes[0]}']] = _result[f'{pid}-01']['tmb']
	else:
		clincDf.loc[_seqSampleIds,[f'TMB-{ignoreGenes[0]}']] = 0
		for pid in clincDf.index.tolist():
			if pid in _result.keys():
				clincDf.loc[pid,[f'TMB-{ignoreGenes[0]}']] = _result[pid]['tmb']
		# For some cptac cohorts pre-processing is needed
		if tissueType in ['brca']:
			clincDf.index = [c[1:] for c in clincDf.index]
		# elif tissueType in ['luad','ucec']:
		# 	print()

	return clincDf.loc[:,[f'TMB-{ignoreGenes[0]}']]
	


def main():

	_inputStudyFolder = '/data/PanCancer/GeneGrouping/MolecularData/PanCancerAtlas/unzipped/brca_tcga_pan_can_atlas_2018'#args.input_study_folder
	#_inputStudyFolder = '/data/PanCancer/HistBiases/data/Molecular/cptac/brca'
	sys.stdout.write(os.path.basename(_inputStudyFolder) + "\t")
	_sampleTmbMap = getTMB(_inputStudyFolder,
						ignoreGenes=['TP53'],
						cohort='tcga')
	print()
if __name__ == '__main__':
	main()