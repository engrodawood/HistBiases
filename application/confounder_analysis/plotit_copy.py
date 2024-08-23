import pandas as pd
import numpy as np
import sys
import math
import os

def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return

def gen_pie(r0,r1,num_gen=20):
    mrks = np.column_stack([
        [0]+np.cos(np.linspace(2*np.pi*r0,2*np.pi*r1,num_gen)).tolist(),
        [0]+np.sin(np.linspace(2*np.pi*r0,2*np.pi*r1,num_gen)).tolist()
    ]
    )
    return mrks

CODEP_PLOT_DICT = {'TOPK':10,# How many genes to show
                   'SIZE':180,# Size of Scatter dot
                    'DONUT_SIZE':10,
                    'SIZE_OFFSET':300,
                   'COLOR':{'MUT':'#BB5566','WT':'#004488','BG':'#FFFFFF',
                            'ALPHA':0.7,'BOX_FACE':'006633','BOX_TICK':'#FFFFFF'}, # Pie charts colors
                   'FONT_SIZE':12,
                    'FIG_SIZE':(12,6)
                   }

OUTPUT_DIR = ''# YOUR REQ
PRED_DIR =f'{OUTPUT_DIR}/Predictions'
CONFOUND_DIR = f'{OUTPUT_DIR}/ConfounderAnalysis/TMB'

OUT_DIR = f'{CONFOUND_DIR}/plot_temp'
mkdir(OUT_DIR)

shift = {'L':1,'M':2,'H':3}
shift_ticks = {'L':'L','M':'M','H':'H'}
tissue = 'colon'

df = pd.read_csv(f'{CONFOUND_DIR}/{tissue}.csv',index_col='Unnamed: 0',delimiter='\t')
df['AUROC'] = [float(d.split('(')[0]) for d in df['AUROC (STD)']]
#doing selection based on min-pvalue
pval_cols = [c for c in df.columns if 'p (' in c]
df = df.iloc[:CODEP_PLOT_DICT['TOPK'],:]
#Grouping values based on varaile of interest
df['PRED'] = [idx.split('_')[0] for idx in df.index]
df = df.sort_values(by='PRED',ascending=True)
df = df.sort_values(by='AUROC',ascending=False)

import matplotlib.pyplot as plt
size = CODEP_PLOT_DICT['SIZE']
base_fontsize = CODEP_PLOT_DICT['FONT_SIZE']
fig,ax = plt.subplots(figsize=CODEP_PLOT_DICT['FIG_SIZE'])
alpha = CODEP_PLOT_DICT['COLOR']['ALPHA']
xticks = []
idx = -1
donuts_coords = []

for key in df.index:
    #coloring base plot based on odd ratio
    key_pairs = key.split('_')
    pred_gene = key_pairs[0]
    auroc = []# AUROC across all folds
    idx = idx+3 if idx>0 else idx+1 
    plots = plt.violinplot(auroc,[idx],showmeans=True,showextrema=False,points=500
                    )
        
    for pc in plots['bodies']:
        pc.set_facecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
        pc.set_edgecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
        pc.set_alpha(0.5)
        pc.set_linewidth(0)
    plots['cmeans'].set_color(CODEP_PLOT_DICT['COLOR']['BOX_TICK'])
    xticks.append([idx,pred_gene])
        
    fields = sorted([p.split('(')[-1].split(')')[0] for p in pval_cols])
    ratio_fields = [f'{f} ratio' for f in fields]
        
    ratios =df.loc[key,ratio_fields].to_numpy()
    from copy import deepcopy
    label_fields = deepcopy(fields)

    for r_idx,r in enumerate([0,1]):
        for fidx,r1 in enumerate(ratios):
            if r_idx>0:# Swapping values in r=1 case
                r0=r1
                r1=r
                color = CODEP_PLOT_DICT['COLOR']['WT']
            else:
                r0=r
                color = CODEP_PLOT_DICT['COLOR']['MUT']
            sidx = shift[fields[fidx]]
            ax.scatter(x = (idx+sidx),y=df.loc[key,f'AUROC ({fields[fidx]})'],c=color,
                        s=size,linewidths=1.5,
                        marker=gen_pie(r0,r1),
                        alpha=alpha
                        )
    for skidx,skey in enumerate(shift.keys()):
        n_counts = df.loc[key,[f'{skey}-{v}' for v in ['WT','MUT']]].sum()*100
        pval =df.loc[key,f'p ({skey})']
        donuts_coords.append([idx+shift[skey],df.loc[key,f'AUROC ({skey})'],
                                math.floor(n_counts) if skidx>0 else math.ceil(n_counts),
                                0 if pval<0.05 else 1
                            ])
        xticks.append([idx+shift[skey],f'{skey}'])
    
    idx = idx+len(shift)
#Coverting pie to donusts
donuts_coords = np.array(donuts_coords)
ax.scatter(donuts_coords[:,0],donuts_coords[:,1],s=CODEP_PLOT_DICT['DONUT_SIZE'],marker='o',c='w')

xticks = np.array(xticks)
d_xticks = {'ticks':xticks[:,0].astype('int'),
            'labels':['All' if tk not in shift.keys() else shift_ticks[tk] for tk in xticks[:,1]],
        #   'rotation':75
}

for cidx in range(donuts_coords.shape[0]):
    xc,yc,counts,pval = donuts_coords[cidx]
    pval = '' if pval>0.0 else '*'
    ax.annotate(int(counts),xy=(xc,yc),
                xytext=(xc,yc-0.032),
                fontsize=8, ha='center', va='center'  
                            )
    ax.annotate(pval,xy=(xc,yc),
                xytext=(xc,yc+0.02),
                fontsize=8, ha='center', va='center'  
                            )
yticks = {'ticks':[0.1,0.20,0.3,0.40,0.50,0.6,0.70,0.8,0.90,1.0],
        'labels':[f'{v:.2f}' for v in [0.1,0.20,0.30,0.40,0.50,0.6,0.70,0.8,0.90,1.0]],'fontsize':base_fontsize-2}
fontDict ={'size':base_fontsize}
plt.xticks(**d_xticks)
plt.yticks(**yticks)
plt.ylim(0.1,1.02)
plt.margins(0.01,0.01)
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_visible(False)
plt.ylabel('AUROC',fontdict=fontDict)

# Drawing vertical lines
offset = 1.5
nxticks = xticks[:,0].astype('int')
annotMinor = np.hstack((
    xticks[xticks[:,-1]==list(shift_ticks.keys())[0]][:,0].astype('int')[:,np.newaxis],
    xticks[xticks[:,-1]==list(shift_ticks.keys())[-1]][:,0].astype('int')[:,np.newaxis]
))

genes = df['PRED'].tolist()

for ids,(pos,lbl) in enumerate(xticks):
    if int(pos)<offset:
        plots = plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
        continue
    if lbl in shift_ticks:
        plots =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
    else:
        plots =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
        plots = plt.axvline(x=int(pos)-offset,ymin=0,ymax=1,linewidth=1,color='black')

# Annotations
fs = 14
from decimal import Decimal
field = 'Grade'


for ids in range(len(annotMinor)):
    xt= annotMinor[ids]
    x0,x1 = float(xt[0]/(idx+offset)),float(xt[1]/(idx+offset))# CRC shift+0.008
    plt.annotate(field, xy=(x1, -0.052), xytext=(x1, -0.102), xycoords='axes fraction', 
    fontsize=fs-4, ha='center', va='bottom',
    arrowprops=dict(arrowstyle=f'-[, widthB={offset}, lengthB=0.2', lw=1.0, color='black'))# CRC grade offset *0.71
    
    plt.annotate(genes[ids], xy=(x1, -0.11), xytext=(x1, -0.18), xycoords='axes fraction', 
    fontsize=fs, ha='center', va='bottom',
    arrowprops=dict(arrowstyle=f'-[, widthB={offset}, lengthB=0.2', lw=1.0, color='black'))
    
plt.tight_layout()

plt.savefig(f'{OUT_DIR}/{tissue}.png',dpi=600)
