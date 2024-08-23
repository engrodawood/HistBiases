import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from application.project import CODEP_COLORS,TISSUE_TYPES_HIST,CODEP_PLOT_DICT
from application.utils.plot_utils import gen_pie

DIR = '/data/PanCancer/HistBiases/output/ConfounderAnalysis/GRADE'

for tissue in ['brca']:#TISSUE_TYPES_HIST:
    df = pd.read_csv(f'{DIR}/{tissue}.csv',index_col='Unnamed: 0',delimiter='\t')
    df['AUROC'] = [float(d.split('(')[0]) for d in df['AUROC (STD)']]
    #doing selection based on min-pvalue
    pval_cols = [c for c in df.columns if 'p (' in c]
    df['MINP'] = df.loc[:,pval_cols].min(1).tolist()
    df = df.sort_values(by='MINP',ascending=True)
    df = df.iloc[:CODEP_PLOT_DICT['TOPK'],:]
    #Grouping values based on varaile of interest
    df['VOI'] = [idx.split('_')[0] for idx in df.index]
    df = df.sort_values(by='VOI',ascending=True)
    
    import matplotlib.pyplot as plt
    size = CODEP_PLOT_DICT['SIZE']
    base_fontsize = CODEP_PLOT_DICT['FONT_SIZE']
    fig,ax = plt.subplots(figsize=CODEP_PLOT_DICT['FIG_SIZE'])
    x = np.arange(df.shape[0])

    alpha = CODEP_PLOT_DICT['COLOR']['ALPHA']
    for idx,key in enumerate(df.index):
        #coloring base plot based on odd ratio
        tend = df.loc[key,'Tendency']
        plt.scatter(x=x[idx],y=df.loc[key,'AUROC'],c=CODEP_COLORS['high'] if tend>0 else CODEP_COLORS['low'],
                    s=size//5,label='CV',marker='d')
        fields = sorted([p.split('(')[-1].split(')')[0] for p in pval_cols])
        if 'MUT' in fields:
            ratio_fields = [f'{f} +ve ratio' for f in fields]
        else:
            ratio_fields = [f'{f} ratio' for f in fields]
        
        ratios =df.loc[key,ratio_fields].to_numpy()
        from copy import deepcopy
        label_fields = deepcopy(fields )
        for r_idx,r in enumerate([0,1]):
            for fidx,r1 in enumerate(ratios):
                if r_idx>0:# Swapping values in r=1 case
                    r0=r1
                    r1=r
                else:
                    r0=r
                plt.scatter(x=x[idx],y=df.loc[key,f'AUROC ({fields[fidx]})'],c=CODEP_PLOT_DICT['COLOR'][label_fields[fidx]],
                            s=size,label = label_fields[fidx],linewidths=1.5,
                            marker=gen_pie(r0,r1),
                            alpha=alpha
                            )
            label_fields.reverse()
    
    # Coverting the chart into a donut and marking outside grid
    for f in fields:
        plt.scatter(x,df[f'AUROC ({f})'],s=CODEP_PLOT_DICT['DONUT_SIZE'],marker='o',c='w')
        plt.scatter(x,df[f'AUROC ({f})'], marker = 'o', s=(size+CODEP_PLOT_DICT['SIZE_OFFSET']), c="None",
            edgecolors=CODEP_PLOT_DICT['COLOR'][f], linewidths=2,linestyle='dashed',alpha=alpha)

    voi_status = ['MUT','WT']
    offset_dict = {0:-0.15,1:0.10,2:0.20}
    for idx,key in enumerate(df.index):
        for fidx,f in enumerate(fields):
            n_counts = df.loc[key,[f'{f}-{v}' for v in voi_status]].sum()*100
            n_counts = int(np.ceil(n_counts)) if fidx<1 else int(np.floor(n_counts))
            pval =df.loc[key,f'p ({f})']
            plt.annotate(n_counts,xy=(x[idx],df.loc[key,f'AUROC ({f})']),
                            xytext=(x[idx],df.loc[key,f'AUROC ({f})']-0.001),
                            fontsize=8, ha='center', va='center'  
                                        )
            if pval<0.05:
                plt.annotate("*",xy=(x[idx],df.loc[key,f'AUROC ({f})']),
                                xytext=(x[idx]+offset_dict[fidx],0.10),
                                fontsize=16, ha='left', va='bottom',color=CODEP_PLOT_DICT['COLOR'][f] 
                                            )
    xticks = {'ticks':x,
            'labels':[idx.split('_')[0] for idx in df.index] ,
            'rotation':45
            }
    plt.xticks(**xticks)
    yticks = {'ticks':[0.0,0.2,0.4,0.6,0.8,1.0],
            'labels':[0.0,0.2,0.4,0.6,0.8,1.0],
            }
    plt.yticks(**yticks)

    yticks = {'ticks':[0,0.2,0.4,0.6,0.8,1.0],'labels':[f'{v:.2f}' for v in [0,0.2,0.4,0.6,0.8,1.0]],'fontsize':base_fontsize-2}
    fontDict ={'size':base_fontsize}
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    plt.ylabel('AUROC',fontdict=fontDict)

    # Drawing horizontal dotted line
    for i in x:
        plt.axvline(x=i+0.5,ymin=0,ymax=1,linestyle='dotted',linewidth=0.2,color='black')


    plt.tight_layout()
    # plt.legend()
    plt.savefig(f'{tissue}_grade.png')
