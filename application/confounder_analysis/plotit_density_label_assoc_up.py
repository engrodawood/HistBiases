import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from application.project import CODEP_COLORS,TISSUE_TYPES_HIST,CODEP_PLOT_DICT,OUTPUT_DIR,get_experiments,SORT_ORDER_ASSOC
from application.utils.plot_utils import gen_pie,mkdir,annotate
from application.utils.performance_eval import auroc_folds_stats
from application.misc.meta import COHORT_GENES_AUC,dict_label_assoc
import math


PRED_DIR =f'{OUTPUT_DIR}/Predictions'
CONFOUND_DIR = f'{OUTPUT_DIR}/ConfounderAnalysis/labels_assoc_cv_fixed'
experiments = get_experiments()
experiments_rev = dict(zip(experiments.values(),experiments.keys()))

OUT_DIR = f'{CONFOUND_DIR}/plot'
mkdir(OUT_DIR)

markers = {'ER':{'WT':'-ve','MUT':'+ve'},
           'PR':{'WT':'-ve','MUT':'+ve'},
           'MSI':{'WT':'Low','MUT':'High'}
           }

shift = {'MUT':1,'WT':2}
cohort = 'tcga'
for tissue in ['ucec']:
    df = pd.read_csv(f'{CONFOUND_DIR}/{cohort}_{tissue}.csv',index_col='Unnamed: 0')
    df['AUROC'] = [float(d.split('(')[0]) for d in df['AUROC (STD)']]
    #doing selection based on min-pvalue
    pval_cols = [c for c in df.columns if 'p (' in c]
    df['MINP'] = df.loc[:,pval_cols].min(1).tolist()
    df = df.sort_values(by='MINP',ascending=True)
    # df = df.iloc[:25,:]#CODEP_PLOT_DICT['TOPK']
    #Grouping values based on varaile of interest
    df['PRED'] = [idx.split('_')[0] for idx in df.index]
    df['VOI'] = [idx.split('_')[1] for idx in df.index]
    #Keeping stratification varaibles with minimum p-values
    df = df.sort_values(by='PRED',ascending=True)
    df = df.sort_values(by='AUROC',ascending=False)
    print()
    #Selecting of only best predictors
    #df = df.dropna()
    #selected = ['MSI','CIMP','CINGS','HM']#,'BRAF','APC','PTEN','RNF43']
    selected = list(dict_label_assoc[cohort][tissue].keys())
    #selected = ['MSI','BRAF','APC','PTEN','RNF43']
    df = df[df.PRED.isin(selected)]
    print()
    # TT = df.groupby('STVar')['MINP'].min().sort_values()
    # selected_genes = TT[TT.values<0.05].index.tolist()
    # df = df[df.STVar.isin(selected_genes)]
    
    
    # Another layer of sorting to bring the best cases forward
    if tissue in SORT_ORDER_ASSOC:
        df['SORT'] = 0
        for idx, mrkr in enumerate(selected):
            df.loc[df.PRED==mrkr,'SORT'] = idx  
        df.loc[~df.PRED.isin(selected),'SORT'] = 1000
    df.sort_values(by='SORT',ascending=True,inplace=True)
    import matplotlib.pyplot as plt
    # plt.rcParams['text.usetex']=True
    size = CODEP_PLOT_DICT['SIZE']
    fsize = CODEP_PLOT_DICT['FONT_SIZE']
    fig,ax = plt.subplots(figsize=CODEP_PLOT_DICT['FIG_SIZE'][f'{cohort}_{tissue}'])
    alpha = CODEP_PLOT_DICT['COLOR']['ALPHA']
    xticks = []
    idx = -1
    done = []
    donuts_coords = []
    pth = 0.5#0.001

    for key in df.index:
        #coloring base plot based on odd ratio
        tend = df.loc[key,'Tendency']
        key_pairs = key.split('_')
        pred_gene,voi,exp = key_pairs[-3],key_pairs[-2],experiments_rev[key_pairs[-1]]
        #if df.loc[key]['MINP']>=pth:continue
        if voi not in dict_label_assoc[cohort][tissue][pred_gene]:continue
        if pred_gene not in done:
            predDf = pd.read_csv(f'{PRED_DIR}/tcga_{tissue}_{exp}.csv')
            auroc = auroc_folds_stats(predDf,
                                            [pred_gene]).to_numpy().ravel()
            idx = idx+3 if idx>0 else idx+1 
            plots = plt.violinplot(auroc,[idx],showmeans=True,showextrema=False,points=500
                            )
            # plt.boxplot(auroc,positions=[idx])
            
            for pc in plots['bodies']:
                # m = np.mean(pc.get_paths()[0].vertices[:,0])
                # pc.get_paths()[0].vertices[:,0] = np.clip(pc.get_paths()[0].vertices[:,0],m,np.inf)
                #pc.set_facecolor(CODEP_COLORS['high'] if tend>0 else CODEP_COLORS['low'])
                pc.set_facecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
                pc.set_edgecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
                pc.set_alpha(0.5)
                # pc.set_linewidth(10)
            plots['cmeans'].set_color(CODEP_PLOT_DICT['COLOR']['BOX_TICK'])
            xticks.append([idx,pred_gene])
            done.append(pred_gene)
        
        
        fields = sorted([p.split('(')[-1].split(')')[0] for p in pval_cols])
        if 'MUT' in fields:
            ratio_fields = [f'{f} +ve ratio' for f in fields]
        else:
            ratio_fields = [f'{f} ratio' for f in fields]
        
        ratios =df.loc[key,ratio_fields].to_numpy()
        from copy import deepcopy
        label_fields = deepcopy(fields)

        # Adding some extra space at start of sub groupds
        idx = idx+1
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
            xticks.append([idx+shift[skey],f'{voi} {skey}'])
      
        idx = idx+np.max(list(shift.values()))
    #Coverting pie to donusts
    donuts_coords = np.array(donuts_coords)
    ax.scatter(donuts_coords[:,0],donuts_coords[:,1],s=CODEP_PLOT_DICT['DONUT_SIZE'],marker='o',c='w')
    offset_dict = {'WT':0.2,'MUT':-0.2}

    xlabels = []
    for _,tk in xticks:
        g = tk.split(' ')[-1]
        if g not in shift.keys():
            xlabels.append('All')
        elif g in shift.keys() and tk.split(' ')[0] in markers:
            xlabels.append(markers[tk.split(' ')[0]][g])
        else:
            xlabels.append(g)

    d_xticks = {'ticks':[r[0]+offset_dict[r[1].split(' ')[-1]] if r[1].split(' ')[-1] in offset_dict.keys()
                          else float(r[0]) for r in xticks],
              'labels':xlabels,#['All' if tk.split(' ')[-1] not in shift.keys() else tk.split(' ')[-1] for _,tk in xticks],
              #'rotation':75
              'fontsize':(fsize-5)
    }
    
    # for cidx in range(donuts_coords.shape[0]):
    #     xc,yc,counts,pval = donuts_coords[cidx]
    #     pval = '' if pval>0.0 else '*'
    #     text = r'${:.0f}\%$'.format(counts)
        
    #     text_offset = -0.25 if cidx % 2==0 else 0.20
    #     ax.annotate(text,xy=(xc+text_offset,yc),
    #                 xytext=(xc+text_offset,yc-0.046),
    #                 fontsize=(fsize-6), ha='center', va='center'  
    #                             )
    #     ax.annotate(pval,xy=(xc,yc),
    #                 xytext=(xc,yc+0.033),
    #                 fontsize=(fsize-2), ha='center', va='center'  
    #    0.10,0.20,0.30,0.4,                         )
    #ytk = [0.50,0.6,0.70,0.8,0.90,1.0]
    #ytk = [0.0,0.20,0.40,0.6,0.8,1.0]
    ytk = [0.1,0.2,0.3,0.4,0.50,0.6,0.70,0.8,0.90,1.0]
    yticks = {'ticks':ytk,
            'labels':[f'{v:.2f}' for v in ytk],'fontsize':fsize-3}
    plt.xticks(**d_xticks)
    plt.yticks(**yticks)
    plt.ylim(0.05,1.03)
    plt.margins(0.02,0.02)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    plt.ylabel('AUROC',fontsize=fsize)

    # CRC grade offset *0.71
    plots = None
    prev = None
    label = None
    offset = 1.5
    for ids,(pos,lbl) in enumerate(xticks):
        if int(pos)<offset:
            label = lbl
            plots = plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
            continue
        if len(lbl.split(' '))>1:
            plots =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
        else:
            plots =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
            plots = plt.axvline(x=int(pos)-offset,ymin=0,ymax=1,linewidth=1,color='black')
            x_loc = plots.get_data()[0][0]/idx
            if prev:
                dsf = (x_loc-prev)/2
            else:
                dsf = x_loc/2
            annotate(label,dsf*50,(x_loc-dsf),fsize-1,
                        -0.12,-0.19)
            prev = x_loc
            label = lbl
    
    # Final annotation
    x_loc = 1
    dsf = (x_loc-prev)/2
    annotate(label,dsf*60,(x_loc-dsf),fsize-1,
                        -0.12,-0.19)
    
    # smx,smy = -0.015,-0.10
    # plt.annotate('Stratification Var:',fontweight='bold', xy=(smx, smy), xytext=(smx, smy), xycoords='axes fraction', 
    #         fontsize=fsize-4, ha='center', va='bottom', color='black')
    
    # lmx,lmy = -0.015,-0.18
    # plt.annotate('Prediction target:',fontweight='bold', xy=(lmx, lmy), xytext=(lmx, lmy), xycoords='axes fraction', 
    #         fontsize=fsize-4, ha='center', va='bottom', color='black')
    
    from decimal import Decimal
    offset = 0.75
    for ids in range(1,len(xticks)):
        x1,text= xticks[ids]
        x1 = float(x1)
        if 'WT' in text:
            x1 = float(x1/(idx+offset))# CRC shift+0.008
            annotate(text,1.5,x1,fsize-3,
                     -0.052,-0.102)

    plt.tight_layout()
    plt.xlim(-1,idx+1)
    plt.grid(axis='y',linestyle='--',linewidth=0.1)

    plt.savefig(f'{OUT_DIR}/{cohort}_{tissue}_noleg.png',dpi=600)
