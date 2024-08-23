# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 19:58:27 2022

@authors: Dawood/ Fayyaz
"""
import pandas as pd
import numpy as np
from  statsmodels.stats.multitest import multipletests
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from application.project import PROJECT_DIR,DATA_DIR,WORKSPACE_DIR,OUTPUT_DIR
from application.utils.utilIO import mkdir
from application.utils.statistical_tests import codependenceTest
from application.utils.data_utils import load_histology,load_molecular
from application.misc.meta import COHORT_GENES,MOL_MARKERS_CODEP

import pandas as pd
# from application.utils.utilIO import mkdir
from pathlib import Path
from application.project import CODEP_COLORS,BIOMARKER_TYPE_COLOR_CONFIG

EXPERIMENT = 'LabelCodependence'
DATA_IN = f'{DATA_DIR}/MUT'
OUT_DIR = f'{OUTPUT_DIR}/{EXPERIMENT}/'

#load data
test='fisher'
pcut = 0.05
orderGene = False
showSelected=False
reorder =False
from glob import glob
files = glob(os.path.join(DATA_IN,'*_b.csv'))
for file in files:
    if 'cptac' in file:continue # Skipping CPTAC as very few samples
    MUTDf = pd.read_csv(file,index_col='Patient ID')
    tissue = Path(file).stem.split('_')[1].strip()
    cohort = Path(file).stem.split('_')[0].strip()
    if orderGene:
        genes = [f.strip() for f in COHORT_GENES[tissue].split(',')]
    else:
        genes = MUTDf.columns.tolist()
        gDict = dict(zip(genes,[BIOMARKER_TYPE_COLOR_CONFIG[0]]*len(genes)))

    if f'{cohort}_{tissue}' in MOL_MARKERS_CODEP.keys():
        MOLDf = load_molecular(cohort=cohort,
                               tissue=tissue)
        if cohort in ['tcga','metabric','dfci'] and tissue in ['brca','colon','ucec']:
            HISTDf = load_histology(
                        tissue=tissue,
                        cohort=cohort
                        )
            MOLDf = MOLDf.join(HISTDf)
            if tissue in ['brca','ucec']:#mapping grade into 2
                MOLDf['Grade'].replace({1:0,2:0,3:1},inplace=True)
                print()
            MUTDf = MOLDf.loc[:,genes+MOL_MARKERS_CODEP[f'{cohort}_{tissue}']+['Grade']]
            marker_types = [BIOMARKER_TYPE_COLOR_CONFIG[0]]*len(genes)+[BIOMARKER_TYPE_COLOR_CONFIG[1]]*len(MOL_MARKERS_CODEP[f'{cohort}_{tissue}'])+[BIOMARKER_TYPE_COLOR_CONFIG[2]]
            gDict = dict(zip(MUTDf.columns.tolist(),marker_types))
        else:
            MUTDf = MOLDf.loc[:,genes+MOL_MARKERS_CODEP[f'{cohort}_{tissue}']]
            marker_types = [BIOMARKER_TYPE_COLOR_CONFIG[0]]*len(genes)+[BIOMARKER_TYPE_COLOR_CONFIG[1]]*len(MOL_MARKERS_CODEP[f'{cohort}_{tissue}'])
            gDict = dict(zip(MUTDf.columns.tolist(),marker_types))

    #perform the chi squared test for all pairs of genes
    P = {}
    Odds = {}
    genes = list(gDict.keys())
    for idx,g1 in enumerate(genes):
        for g2 in genes[idx+1:]:
            pval,log_odds = codependenceTest(df = MUTDf,
                               pairs = [g1,g2],
                               test=test)
            P[(g1,g2)] = pval
            Odds[(g1,g2)] = log_odds

    #perform multiple hypothesis correction
    reject,pvc,_,_ = multipletests(list(P.values()), alpha = pcut,
                                    method = 'fdr_bh' )
    
    # Sometime the value after correction are set to zero.
    # In that case original p-value is picked. Would not matter
    # much given the result is already statistically significant.
    pvc[pvc==0] =np.array(list(P.values()))[pvc==0]
    # collate the corrected p-values of all pairs of genes into a dataframe
    Pc = dict(zip(P.keys(),pvc))
    # import pdb; pdb.set_trace()

    M = np.zeros((len(genes),len(genes)))*np.nan
    O = np.zeros((len(genes),len(genes)))*np.nan
    for idx1,g1 in enumerate(genes):
        for idx2,g2 in enumerate(genes):
            if g1==g2:
                M[idx1,idx2] = 1
                continue
            try:
                M[idx1,idx2] = Pc[(g1,g2)]
                O[idx1,idx2] = Odds[(g1,g2)]
            except KeyError:
                M[idx1,idx2] = Pc[(g2,g1)]
                O[idx1,idx2] = Odds[(g2,g1)]

    # Adding counts to gene name 
    #genes = [f'{gene} [{sum(MUTDf[gene]==1)/MUTDf.shape[0]:.2f}]'.replace('0.','.') for gene in genes]       
    Mdf = pd.DataFrame(M,columns = genes, index = genes)
    data = -np.log10(Mdf)
    # Seleting matrix entries based on Significance thresold
    sssP,ssP,sP,iP = Mdf<0.001,(Mdf<0.01)*(Mdf>=0.001),(Mdf<pcut)*(Mdf>=0.01),Mdf>=pcut
    Mdf[iP] = ''
    Mdf[sssP] = '***' # setting to zero
    Mdf[ssP] = '**'
    Mdf[sP] = '*'
    
    #Matrix of Odd ratios
    Odf = pd.DataFrame(O,columns = genes, index = genes)
  
    if reorder:
        # Not sorting in case of comparing the test results
        ridx = np.argsort(-np.nansum(data,axis=1))
        data = data.iloc[ridx,:]
        cidx = np.argsort(-np.nansum(data,axis=0))
        data = data.iloc[:,cidx]

        Odf = Odf.loc[data.index.tolist(),data.columns.tolist()]
        Mdf = Mdf.loc[data.index.tolist(),data.columns.tolist()]
    if 'ucec' in file:
        fig, axs = plt.subplots(2,figsize=(14,14),gridspec_kw={'height_ratios': [2, 12],'wspace':0},sharex=True)
        tick_fsize = 14
        annot_fsize = 8
        row_width = 0.018
        row_margin = 0.020
        y_offset = 0.02
    else:
        fig, axs = plt.subplots(2,figsize=(7,7),gridspec_kw={'height_ratios': [2, 12],'wspace':0.0},sharex=True)
        tick_fsize = 12
        annot_fsize = 8
        row_width = 0.04
        row_margin = 0.045
        y_offset = 0.02

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(Odf, dtype=bool))
    
     # Creating Seaborn Heatmap
    import matplotlib.colors
    norm = matplotlib.colors.Normalize(-3,3)
    colors = [[norm(-3.0), CODEP_COLORS['low']],
            # [norm(-0.3), "#DDDDDD"],
            [norm(0), CODEP_COLORS['medium']],
            [norm( 3.0), CODEP_COLORS['high']]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    import seaborn as sns
    # cbar_ax = fig.add_axes([.70, .2, .025, .3])
    sm = sns.heatmap(Odf.iloc[::-1],mask=mask[::-1],ax=axs[1],cbar=False,cmap=cmap,#cbar_ax = cbar_ax,
                    vmin = -3,vmax=3,center=0,linewidths=.5,
                    annot = Mdf.iloc[::-1],fmt='s',square=True,#,cbar_kws={"shrink": .5,'label':'Log2 Odds Ratio'},
                    annot_kws ={'size':annot_fsize,'color':'black'}
    )

    colors = list(gDict.values())[::-1]
    for i, color in enumerate(colors):
        ax = axs[1]
        if i<len(colors)-1:
            height = 0.95 if color!=colors[i+1] else 1
        else:
            height = 1
        ax.add_patch(plt.Rectangle(xy=(-row_margin, i+y_offset), width=row_width, height=height, color=color, lw=0,
                                transform=ax.get_yaxis_transform(), clip_on=False))
        prev = color

    axs[1].tick_params(axis='x', which='major', labelsize=tick_fsize, labelbottom = False, bottom=False, top = True, labeltop=False,
    left=False,right=False,labelrotation=90,pad=0)
    axs[1].tick_params(axis='y', which='major',labelrotation=0,pad=20,length=0)

    for label in axs[1].get_yticklabels():
        label.set_ha('right')
    pad = 0.15 if 'brca' in file else 0.15
    pad2 = 65 if 'brca' in file else 55
    from mpl_toolkits.axes_grid import make_axes_locatable
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("left", size="12%", pad=pad)
    cax.remove()
    cax = divider.append_axes("right", size="12%", pad=pad)
    cax.remove()
    axs[0].grid(True,linewidth=0.25,linestyle='--',zorder=0,color='#DDDDDD')
    axs[0].bar(axs[1].get_xticks(),[(sum(MUTDf[g]==1)/MUTDf.shape[0]) for g in genes],color='#CCDDAA',alpha=0.8,zorder=2)
    axs[0].tick_params(axis='x',which='major',bottom=True, top = False, labelbottom=True,pad=pad2,
    left=True,right=False)
    axs[0].tick_params(axis='x', which='major',labelrotation=90)
    axs[0].set_ylabel(r'Positive (%)')

    for label in axs[0].get_xticklabels():
        label.set_va('bottom')

    axs[0].set_ylim(0,1)
    yticks = {'ticks':[0,0.5,1]}
    axs[0].set_yticks(**yticks)
    axs[0].set_xticklabels(genes)
    

    plot_dir =f'{OUT_DIR}/plots_TEMP'
    mkdir(plot_dir)

    csv_dir =f'{OUT_DIR}/csv'
    mkdir(csv_dir)
    outname = Path(file).stem[:-2]
    Mdf.to_csv(f'{csv_dir}/{outname}_{test}.csv')
    plt.tight_layout()
    plt.gca().set_axisbelow(True)
    plt.savefig(f'{plot_dir}/{outname}_{test}.png',dpi=600,bbox_inches='tight')
