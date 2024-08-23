
### Description

- **Histology/**: Contains histological data such as Cancer Grade and other information for two different cohorts:
  - **cptac_ucec/**: Histology data for CPTAC UCEC cohort.
  - **tcga_brca/**: Histology data for TCGA BRCA cohort.
  - **tcga_colon/**: Histology data for TCGA colon cohort from the paper by Alzaid et al, 2024: https://arxiv.org/abs/2405.02040

- **META_WSIs/**: Metadata associated with Whole Slide Images (WSIs).
  - **gdc_sample_sheet_mutation.tsv**: A tab-separated file containing set of TCGA sample analysed in the study.
  - **wsi_meta_mfilled.xlsx**: An Excel file with manually filled MPP information based on tissue source site for the set of slides with missing metadata.
  - **wsi_meta.csv**: A CSV file containing metadata for WSIs.

- **Molecular/**: This directory is intended to contain molecular data, such as receptor status in breast tumours, MSI status of colorectal tumours etc. 

- **MUT/**: This directory is intended to contain mutation data for different cohorts downloaded from cBioportal. For any missing file that is used in the repo but missing in the data folder please download it cBioportal. 

### Source

The data in this directory has been primarily downloaded from [cBioPortal](https://www.cbioportal.org/), a comprehensive platform for exploring, visualizing, and analyzing cancer genomic data.

### Notes

- Ensure that the files are correctly linked and paths are appropriately set when using them in your analysis scripts.
- The data files are assumed to be processed and organized in a manner suitable for the project needs.

For any questions or issues with the data, please refer to the documentation on cBioportal or create an issue and we would be happy to respond.
