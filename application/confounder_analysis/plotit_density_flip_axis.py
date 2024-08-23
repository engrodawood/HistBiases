import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from application.project import CODEP_COLORS,TISSUE_TYPES_HIST,CODEP_PLOT_DICT,OUTPUT_DIR,get_experiments
from application.utils.plot_utils import gen_pie
from application.utils.performance_eval import auroc_folds_stats
import math

PRED_DIR = PRED_DIR =f'{OUTPUT_DIR}/Predictions'
CONFOUND_DIR = f'{OUTPUT_DIR}/ConfounderAnalysis/labels_assoc_cv'
experiments = get_experiments()
experiments_rev = dict(zip(experiments.values(),experiments.keys()))

markers = {'ER':{'WT':'-ve','MUT':'+ve'},
           'ER':{'WT':'-ve','MUT':'+ve'},
           'MSI':{'WT':'Low','MUT':'High'}
           }
shift = {'MUT':1,'WT':2}
for tissue in ['brca']:#TISSUE_TYPES_HIST:
    df = pd.read_csv(f'{CONFOUND_DIR}/{tissue}.csv',index_col='Unnamed: 0')
    df['AUROC'] = [float(d.split('(')[0]) for d in df['AUROC (STD)']]
    #doing selection based on min-pvalue
    pval_cols = [c for c in df.columns if 'p (' in c]
    df['MINP'] = df.loc[:,pval_cols].min(1).tolist()
    df = df.sort_values(by='MINP',ascending=True)
    df = df.iloc[:CODEP_PLOT_DICT['TOPK'],:]
    #Grouping values based on varaile of interest
    df['PRED'] = [idx.split('_')[1] for idx in df.index]
    df = df.sort_values(by='PRED',ascending=True)
    df = df.sort_values(by='AUROC',ascending=True)
    
    import matplotlib.pyplot as plt
    size = CODEP_PLOT_DICT['SIZE']
    base_fontsize = CODEP_PLOT_DICT['FONT_SIZE']
    fig,ax = plt.subplots(figsize=CODEP_PLOT_DICT['FIG_SIZE'])
    alpha = CODEP_PLOT_DICT['COLOR']['ALPHA']
    xticks = []
    xticks_lines = []

    from statsmodels.graphics.boxplots import violinplot
    idx = -1
    donuts_coords = []
    for key in df.index:
        #coloring base plot based on odd ratio
        tend = df.loc[key,'Tendency']
        key_pairs = key.split('_')
        voi,pred_gene,exp = key_pairs[-3],key_pairs[-2],experiments_rev[key_pairs[-1]]
        if pred_gene not in xticks:
            if idx>0:
                xticks_lines.append(idx)
            idx = idx+1
            predDf = pd.read_csv(f'{PRED_DIR}/tcga_{tissue}_{exp}.csv')
            auroc = auroc_folds_stats(predDf,
                                        [pred_gene]).to_numpy().ravel()
            
            plots = plt.violinplot(auroc,[idx],showmeans=True,showextrema=False,points=500,vert=False
                        )
            
            xticks.append(pred_gene)
            for pc in plots['bodies']:
                m = np.mean(pc.get_paths()[0].vertices[:,0])
                pc.get_paths()[0].vertices[:,0] = np.clip(pc.get_paths()[0].vertices[:,0],m,np.inf)
                pc.set_facecolor('#BB5566')
        fields = sorted([p.split('(')[-1].split(')')[0] for p in pval_cols])
        if 'MUT' in fields:
            ratio_fields = [f'{f} +ve ratio' for f in fields]
        else:
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
                ax.scatter(y = (idx+sidx),x=df.loc[key,f'AUROC ({fields[fidx]})'],c=color,
                            s=size,linewidths=1.5,
                            marker=gen_pie(r0,r1),
                            alpha=alpha
                            )
        for skidx,skey in enumerate(shift.keys()):
            n_counts = df.loc[key,[f'{skey}-{v}' for v in shift.keys()]].sum()*100
            pval =df.loc[key,f'p ({skey})']
            donuts_coords.append([idx+shift[skey],df.loc[key,f'AUROC ({skey})'],
                                  math.floor(n_counts) if skidx>0 else math.ceil(n_counts),
                                 0 if pval<0.05 else 1
                                ])
            if voi in markers:
                xticks.append(f'{voi} {markers[voi][skey]}')
            else:
                xticks.append(f'{voi} {skey}')
      
        idx = idx+len(shift)
    
    #Coverting pie to donusts
    donuts_coords = np.array(donuts_coords)
    ax.scatter(donuts_coords[:,1],donuts_coords[:,0],s=CODEP_PLOT_DICT['DONUT_SIZE'],marker='o',c='w')
    x = np.arange(idx+1)
    xticks = {'ticks':x,
              'labels':xticks,
              'rotation':0
    }
    
    for cidx in range(donuts_coords.shape[0]):
        xc,yc,counts,pval = donuts_coords[cidx]
        xc,yc = yc,xc
        pval = '' if pval>0.0 else '*'
        ax.annotate(int(counts),xy=(xc,yc),
                    xytext=(xc,yc-0.02),
                    fontsize=10, ha='center', va='center'  
                                )
        ax.annotate(pval,xy=(xc,yc),
                    xytext=(xc,yc+0.016),
                    fontsize=10, ha='center', va='center'  
                                )
    # yticks = {'ticks':[0.40,0.50,0.6,0.70,0.8,0.90,1.0],'labels':[f'{v:.2f}' for v in [0.40,0.50,0.6,0.70,0.8,0.90,1.0]],'fontsize':base_fontsize-2}
    # fontDict ={'size':base_fontsize}
    plt.yticks(**xticks)
    # plt.yticks(**yticks)
    # ax.xaxis.label.set_color('black')
    # ax.yaxis.label.set_color('black')
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['top'].set_visible(False)
    # plt.ylabel('AUROC',fontdict=fontDict)
    # # # Drawing horizontal dotted line
    # for i in x:
    #     if i in xticks_lines:
    #         plt.axhline(x=i+0.6,ymin=0,ymax=1,linewidth=0.5,color='black')
    #     else:
    #         plt.axhline(x=i,ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')


    plt.tight_layout()
    # plt.legend()
    plt.savefig(f'{tissue}_FM_HF.png')