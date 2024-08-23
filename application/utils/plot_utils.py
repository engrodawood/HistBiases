import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from application.utils.utilIO import mkdir
from application.project import BAR_PLOT_DICT

def annotate(text,width,x,fsize,yx,yt):
        plt.annotate(text.split(' ')[0], xy=(x, yx), xytext=(x, yt), xycoords='axes fraction', 
            fontsize=fsize, ha='center', va='bottom',
            arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=0.2', lw=1.0, color='black'))

def gen_pie(r0,r1,num_gen=20):
    mrks = np.column_stack([
        [0]+np.cos(np.linspace(2*np.pi*r0,2*np.pi*r1,num_gen)).tolist(),
        [0]+np.sin(np.linspace(2*np.pi*r0,2*np.pi*r1,num_gen)).tolist()
    ]
    )
    return mrks

def comparision_barplot(plotDf,
                        columns=['Ours','SOTA'],
                        base_fontsize=BAR_PLOT_DICT['FONT_SIZE'],
                        save_path = None,
                        shift=0.21,
                        width = 0.40,
                        colors = BAR_PLOT_DICT['COLOR'],
                        Annot=False
                    ):
    """
     plotDf: dataframe with mean and std
     columns: List of columns
     for each columns there should be a correspnding std value
    """
    fig,ax = plt.subplots(figsize=BAR_PLOT_DICT['FIG_SIZE'])
    x = np.arange(plotDf.shape[0])

    for idx,exp in enumerate(columns):
            mStd = plotDf.loc[:,[f'{exp} MEAN',f'{exp} STD']].to_numpy()
            ERROR_BAR = True
            if ERROR_BAR:
                ax.bar((x-shift),mStd[:,0],width=width,xerr=None,yerr=mStd[:,1],
                        color=colors[idx],label=exp)
            else:
                ax.plot(x,mStd[:,0],label=exp)
                ax.fill_between(x,mStd[:,0]-mStd[:,1],mStd[:,0]+mStd[:,1],alpha=0.2)
            shift = shift*-1
    # plt.legend(fontsize=base_fontsize)
    plt.legend(None)
    xticks = {'ticks':x,'labels':plotDf.index.tolist(),'fontsize':base_fontsize-3,'rotation':75}
    yticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    yticks = {'ticks':yticks,'labels':yticks,'fontsize':base_fontsize-3}
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    plt.ylabel('AUROC',fontsize=base_fontsize-1)
    if Annot:
        AnnotDf = plotDf.groupby('Type')['Ours STD'].count()
        AnnotDf = AnnotDf/AnnotDf.sum()
        smys,smye = -0.26,-0.36
        if AnnotDf.shape[0]>1:
             biohalf = AnnotDf['Biomarkers']/2
             annotate('Biomarkers',(biohalf)*38,biohalf,base_fontsize-1,smys,smye)
             muthalf = ((AnnotDf['Mutations']/2)+AnnotDf['Biomarkers'])
             annotate('Mutations',(muthalf)*22,muthalf,base_fontsize-1,smys,smye)
            #  smxm,smxb  = AnnotDf['Mutations'],AnnotDf['Biomarkers']
            #  annotate()
        else:
            annotate('Mutations',(0.5)*39,0.5,base_fontsize-2,smys,smye)

    plt.tight_layout()
    plt.savefig(save_path,dpi=600)


def comparision_barplot_group(plotDf,
                        columns=['Ours','SOTA'],
                        base_fontsize=BAR_PLOT_DICT['FONT_SIZE'],
                        save_path = None,
                        shift=0.21,
                        width = 0.40,
                        colors = BAR_PLOT_DICT['COLOR'],
                        Annot=False,
                        mutProd = 20,
                        bioProd = 38,
                        key='tcga_brca'
                    ):
    """
     plotDf: dataframe with mean and std
     columns: List of columns
     for each columns there should be a correspnding std value
    """
    fig,ax = plt.subplots(figsize=BAR_PLOT_DICT['FIG_SIZE'][key])
    plt.margins(0.015,0.015)
    x = np.arange(plotDf.shape[0])
    bars = np.linspace(-shift,shift,len(columns))
    bin_width = width
    print(len(columns))
    for idx,exp in enumerate(columns):
        mean_ = plotDf.loc[:,[f'{exp} MEAN']].to_numpy()
        cil_ = mean_-plotDf.loc[:,[f'{exp} CIl']].to_numpy()
        cih_ = plotDf.loc[:,[f'{exp} CIh']].to_numpy()-mean_
        # ERROR_BAR = True
        # if ERROR_BAR:
        ax.bar((x+bars[idx]),mean_.ravel(),width=bin_width,xerr=None,yerr=np.hstack((cil_,cih_)).T,capsize=2,
                    color=colors[idx],label=exp)
        # else:
        #     ax.plot(x,mStd[:,0],label=exp)
        #     ax.fill_between(x,mStd[:,0]-mStd[:,1],mStd[:,0]+mStd[:,1],alpha=0.2)
        # shift = shift*-1
    #plt.legend(fontsize=base_fontsize-4)
    #plt.legend(None)
    xticks = {'ticks':x,'labels':plotDf.index.tolist(),'fontsize':base_fontsize-3,'rotation':75}
    yticks = [0,0.2,0.4,0.6,0.8,1.0]
    yticks = {'ticks':yticks,'labels':yticks,'fontsize':base_fontsize-3}
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    plt.ylabel('AUROC',fontsize=base_fontsize-1)
    if Annot:
        AnnotDf = plotDf.groupby('Type')[f'{exp} MEAN'].count()
        AnnotDf = AnnotDf/AnnotDf.sum()
        #smys,smye = -0.27,-0.37
        #smys,smye = -0.34,-0.44
        smys,smye = -0.22,-0.32 #for ABCTB
        #smys,smye = -0.58,-0.72 #for UCEC
        if AnnotDf.shape[0]>1:
             biohalf = AnnotDf['Biomarkers']/2
             annotate('Biomarker',(biohalf)*bioProd,biohalf,base_fontsize-1,smys,smye)
             muthalf = ((AnnotDf['Mutations']/2)+AnnotDf['Biomarkers'])
             annotate('Mutations',(muthalf)*mutProd,muthalf,base_fontsize-1,smys,smye)
            #  smxm,smxb  = AnnotDf['Mutations'],AnnotDf['Biomarkers']
            #  annotate()
        else:
            #annotate('Mutations',(0.5)*mutProd,0.5,base_fontsize-2,smys,smye)
            # for ABCTB case
            annotate('Biomarkers',(0.5)*bioProd,0.5,base_fontsize-2,smys,smye)

    plt.tight_layout()
    plt.savefig(save_path,dpi=600,bbox_inches='tight')


def plot_permutation_test(B=None,
                          F=None,
                          uvals = None,
                          VOI=None,
                          LV = None
                          ):
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science','muted'])
    plt.style.use(['no-latex'])
    plt.rcParams['axes.formatter.use_mathtext']=False
    plt.rcParams['axes.formatter.useoffset'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['figure.raise_window']=False
    plt.rcParams['figure.figsize']=[4.5,3.625]
    from cycler import cycler
    # custom_cycler = (cycler(color=['#999933', '#DDDDDD', '#CC6677']) +
    #              cycler(lw=[2, 0.5, 2]))
    # plt.rcParams['axes.prop_cycle']=custom_cycler
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['xtick.major.top']=False
    plt.rcParams['ytick.major.right']=False
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['font.family']=['Times New Roman']
    # plt.locator_params(numticks=8)
    colors = ['g','r','b']
    bins = np.linspace(0.5,1.0,100)
    plt.figure()
    for i,f in enumerate(F):
        #plt.figure()    
        plt.hist(B[:,i],bins=bins,density = True, alpha=0.7,histtype = 'step',color = colors[i])
        xticks = {'ticks':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],'labels':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
        plt.xticks(**xticks)
        labels, locations = plt.yticks()
        yticks = {'ticks':labels,'labels':['' for i in labels]}
        plt.yticks(**yticks)   
        
    for i,f in enumerate(F): plt.scatter(f,1.0,marker = 'o',color = colors[i])
    plt.legend(list(uvals)*2)
    #plt.title(f'background density (lines) and observed values (dots)')
    #plt.xlabel(f'{cohort.upper()}, VOI: {VOI.split("_")[-1]}, Label: {LV.split("_")[-1]}, p {p}, EXP: {experiments[exp]}')
    plt.xlabel('AUROC', fontdict={'size':14}) 
    plt.ylabel('Probability',fontdict={'size':14})
    # plot_dir =f'{OUT_DIR}/{cohort}/'
    # mkdir(plot_dir)

    # plt.savefig(f'{plot_dir}{cohort}_{VOI}_{LV}_{experiments[exp]}_dist.png')
    # plt.close()
    # plt.figure();plt.boxplot([score[voi==v] for v in uvals]); 
    # plt.xlabel(f'{cohort.upper()}, VOI: {VOI.split("_")[-1]}, Label: {LV.split("_")[-1]}, p {p}, EXP: {experiments[exp]}');
    # plt.ylabel('Distribution of scores within group')#fix the ticks
    # plt.savefig(f'{plot_dir}{cohort}_{VOI}_{LV}_{experiments[exp]}_box.png')
    # plt.close()