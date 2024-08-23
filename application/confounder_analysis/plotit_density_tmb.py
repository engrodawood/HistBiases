import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from application.project import CODEP_COLORS,TISSUE_TYPES_HIST,CODEP_PLOT_DICT,OUTPUT_DIR,get_experiments
from application.misc.meta import COHORT_GENES_AUC,COHORT_GENES_TMB
from application.utils.plot_utils import gen_pie,mkdir
from application.utils.performance_eval import auroc_folds_stats,bootstrap_runs_mout
import math


PRED_DIR =f'{OUTPUT_DIR}/Predictions'
CONFOUND_DIR = f'{OUTPUT_DIR}/ConfounderAnalysis/TMB_HL'
experiments = get_experiments()
experiments_rev = dict(zip(experiments.values(),experiments.keys()))

OUT_DIR = f'{CONFOUND_DIR}/plots'
mkdir(OUT_DIR)

markers = {'ER':{'WT':'-ve','MUT':'+ve'},
           'ER':{'WT':'-ve','MUT':'+ve'},
           'MSI':{'WT':'Low','MUT':'High'}
           }
shift = {'L':1,'M':2,'H':3}
shift = {'L':1,'H':2}
shift_ticks = {'L':'L','M':'M','H':'H'}
#shift_ticks = {'L':'0','M':'1','H':'2'}
shift_ticks = {'L':'0','H':'1'}
cohort = 'cptac'
for tissue in ['ucec']:#TISSUE_TYPES_HIST:
    df = pd.read_csv(f'{CONFOUND_DIR}/{cohort}_{tissue}_nan_fixed.csv')
    df['AUROC'] = [float(d.split('(')[0]) for d in df['AUROC (STD)']]
    #doing selection based on min-pvalue
    pval_cols = [c for c in df.columns if 'p (' in c]
    df['MINP'] = df.loc[:,pval_cols].min(1).tolist()
    # df = df.sort_values(by='MINP',ascending=True)
    #df = df.iloc[:CODEP_PLOT_DICT['TOPK'],:]
    #Grouping values based on varaile of interest
    df['PRED'] = [idx.split('_')[0] for idx in df['Unnamed: 0']]
    df.set_index('PRED',inplace=True)
    sel_genes = ['PTEN', 'TP53', 'CTNNB1', 'ARID1A','RNF43']
    #sel_genes = ['BRAF', 'TP53', 'APC', 'KRAS','PIK3CA']
    df = df.loc[sel_genes,:]
    # Reseting the index
    df = df.reset_index()
    df.set_index('Unnamed: 0',inplace=True)
    # df = df.sort_values(by='PRED',ascending=True)
    # df = df.sort_values(by='AUROC',ascending=False)

    import matplotlib.pyplot as plt
    size = CODEP_PLOT_DICT['SIZE']
    fsize = CODEP_PLOT_DICT['FONT_SIZE']-1
    fig,ax = plt.subplots(figsize=CODEP_PLOT_DICT['FIG_SIZE_TMB'][f'{cohort}_{tissue}'])
    alpha = CODEP_PLOT_DICT['COLOR']['ALPHA']
    xticks = []
    h1_ticks,h2_ticks = [],[]
    idx = -1
    donuts_coords = []

    for key in df.index:
        #coloring base plot based on odd ratio
        tend = df.loc[key,'Tendency']
        key_pairs = key.split('_')
        pred_gene,exp = key_pairs[0],experiments_rev[key_pairs[-1]]
        h1t,h2t= [],[]

        predDf = pd.read_csv(f'{PRED_DIR}/{cohort}_{tissue}_{exp}.csv')
        auroc = bootstrap_runs_mout(predDf,
                                        [pred_gene],
                                        runs=100
         ).dropna(0).to_numpy().ravel()
        idx = idx+3 if idx>0 else idx+1 
        plots = plt.violinplot(auroc,[idx],showmeans=True,showextrema=False,points=500
                        )
        
        for pc in plots['bodies']:
            # m = np.mean(pc.get_paths()[0].vertices[:,0])
            # pc.get_paths()[0].vertices[:,0] = np.clip(pc.get_paths()[0].vertices[:,0],m,np.inf)
            #pc.set_facecolor(CODEP_COLORS['high'] if tend>0 else CODEP_COLORS['low'])
            pc.set_facecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
            pc.set_edgecolor(CODEP_PLOT_DICT['COLOR']['BOX_FACE'])
            pc.set_alpha(0.5)
            pc.set_linewidth(0)
        plots['cmeans'].set_color(CODEP_PLOT_DICT['COLOR']['BOX_TICK'])
        xticks.append([idx,pred_gene])
           

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
            xticks.append([idx+shift[skey],f'{skey}'])
      
        idx = idx+len(shift)
    #Coverting pie to donusts
    donuts_coords = np.array(donuts_coords)
    ax.scatter(donuts_coords[:,0],donuts_coords[:,1],s=CODEP_PLOT_DICT['DONUT_SIZE'],marker='o',c='w')

    xticks = np.array(xticks)
    d_xticks = {'ticks':xticks[:,0].astype('int'),
              'labels':['All' if tk not in shift.keys() else shift_ticks[tk] for tk in xticks[:,1]],
              'fontsize':(fsize-3)
            #   'rotation':75
    }
    #offset_shift = {0:-0.3,1:0.1,2:0.5} # for BRCA
    # offset_shift = {0:-0.45,1:0.5,2:0.5}
    # for cidx in range(donuts_coords.shape[0]):
    #     xc,yc,counts,pval = donuts_coords[cidx]
    #     pval = '' if pval>0.0 else '*'
    #     text = r'${:.0f}\%$'.format(counts)
    #     #text_offset = -0.3 if cidx % 3==0 else 0
    #     text_offset = offset_shift[cidx%2]
        
    #     ax.annotate(text,xy=(xc+text_offset,yc),
    #                 xytext=(xc+text_offset,yc-0.062),
    #                 fontsize=(fsize-4), ha='center', va='center'  
    #                             )
    #     ax.annotate(pval,xy=(xc,yc),
    #                 xytext=(xc,yc+0.042),
    #                 fontsize=(fsize-2), ha='center', va='center'  
    #                             )
    # yticks = {'ticks':[0.0,0.1,0.20,0.3,0.40,0.50,0.6,0.70,0.8,0.90,1.0],
    #         'labels':[f'{v:.2f}' for v in [0.0,0.1,0.20,0.30,0.40,0.50,0.6,0.70,0.8,0.90,1.0]],'fontsize':fsize-3}
    yticks = {'ticks':[0.2,0.3,0.40,0.50,0.6,0.70,0.8,0.90,1.0],
            'labels':[f'{v:.2f}' for v in [0.2,0.30,0.40,0.50,0.6,0.70,0.8,0.90,1.0]],'fontsize':fsize-3}
    fontDict ={'size':fsize-3}
    plt.xticks(**d_xticks)
    plt.yticks(**yticks)
    plt.ylim(0.15,1.04)
    plt.margins(0.02,0.02)
    #plt.margins(0.001,0.00)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    plt.ylabel('AUROC',fontsize=fsize)

    # Drawing vertical lines
    offset = 1.5
    nxticks = xticks[:,0].astype('int')
    annotMinor = np.hstack((
        xticks[xticks[:,-1]==list(shift_ticks.keys())[0]][:,0].astype('int')[:,np.newaxis],
        xticks[xticks[:,-1]==list(shift_ticks.keys())[-1]][:,0].astype('int')[:,np.newaxis]
    ))
 
    annotMajor = []
    genes = df['PRED'].tolist()
    for ids in range(len(genes)-1):
        skey,nkey = genes[ids],genes[ids+1]
        annotMajor.append([int(xticks[xticks[:,-1]==skey][0][0]),
                   int(xticks[xticks[:,-1]==nkey][0][0]),
                   skey])
    plots = None
    annotM = []
    for ids,(pos,lbl) in enumerate(xticks):
        if int(pos)<offset:
            plots = plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
            continue
        if lbl in shift_ticks:
            line =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
            annotM.append(line.get_xdata())
        else:
            plots =plt.axvline(x=int(pos),ymin=0,ymax=1,linestyle='dotted',linewidth=0.1,color='black')
            plots = plt.axvline(x=int(pos)-offset,ymin=0,ymax=1,linewidth=1,color='black')
    
   
    from decimal import Decimal
    field = 'TMB'
    #x_off,sfactor = 0.02,2.45 for luad
    #x_off,sfactor = 0.016,1.65 colon
    #x_off,xs_off,sfactor,ssfactor =0.016,0.002,2.45,1.25# ssfactor to be set only for luad 1.25
    #x_off,xs_off,sfactor,ssfactor =0.016,0.002,1.8,1.4# BRCA grade TCGA
    #x_off,xs_off,sfactor,ssfactor =0.016,0.002,2,1.7# BRCA grade ABCTB
    x_off,xs_off,sfactor,ssfactor =0.015,0,1.25,0.50 # colon grade

    offset = 2
    xsshift,xeshift = 1.95,1.82
    xtt = np.linspace((annotMinor.min()+xsshift)/idx,(annotMinor.max()-xeshift)/idx,annotMinor.shape[0])
    for ids in range(len(annotMinor)):
        if genes[ids] not in ['MSI','ER','PR','HER2']:
            text = r"$\mathrm{TMB}_{\widetilde{" + genes[ids] + "}}$" if field=='TMB' else 'Grade'
        else:
            text = 'TMB' if field=='TMB' else 'Grade'
        xt= annotMinor[ids]
        print(xt)
        #x0,x1 = float(xt[0]/(idx+offset)),float(xt[1]/(idx+offset))# CRC shift+0.008
        #x0,x1 = float(xt[0]/(idx+offset)),float(xt[1]/(idx+offset))# CRC shift+0.008
        x = (xt[0]/idx+xt[1]/idx)/2
        x0,x1 =xtt[ids],xtt[ids]# float(x),float(x)

        plt.annotate(text, xy=(x1-xs_off, -0.071), xytext=(x1-xs_off, -0.115), xycoords='axes fraction', 
        fontsize=fsize-4, ha='center', va='bottom',
        arrowprops=dict(arrowstyle=f'-[, widthB={offset*ssfactor}, lengthB=0.2', lw=1.0, color='black'))# CRC grade offset *0.71
        
        plt.annotate(genes[ids], xy=(x1-x_off, -0.129), xytext=(x1-x_off, -0.186), xycoords='axes fraction', 
        fontsize=fsize-1, ha='center', va='bottom',
        arrowprops=dict(arrowstyle=f'-[, widthB={offset*sfactor}, lengthB=0.2', lw=1.0, color='black'))
    
    # Adding starting labels
        
    # smx,smy = -0.04,-0.11
    # plt.annotate('Stratification Var:',fontweight='bold', xy=(smx, smy), xytext=(smx, smy), xycoords='axes fraction', 
    #         fontsize=fsize-4, ha='center', va='bottom', color='black')
    
    # lmx,lmy = -0.04,-0.174
    # plt.annotate('Prediction target:',fontweight='bold', xy=(lmx, lmy), xytext=(lmx, lmy), xycoords='axes fraction', 
    #         fontsize=fsize-4, ha='center', va='bottom', color='black')
    plt.tight_layout()
    plt.xlim(-1,idx+1)

    plt.savefig(f'{OUT_DIR}/{cohort}_{tissue}_noleg.png',dpi=600,bbox_inches='tight')
