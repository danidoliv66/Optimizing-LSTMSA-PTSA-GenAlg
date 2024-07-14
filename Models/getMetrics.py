# -*- coding: utf-8 -*-
"""
Get metrics stores by Encoder_Decoder.py
Set these functions in `saveModels script`

ACC per class is not a good metric for ImBalanced
TPR -> Sensitivity
TNR -> Specificity (Not common in papers)

Typically, decrease of ACC + increase of TPR, TNR and F1 is Good for Imbalanced
"""
#%% Import libraries
import pickle
import pandas as pd
import numpy as np
from time import strftime, gmtime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from pretty_confusion_matrix import pp_matrix
pio.renderers.default='browser'
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

#%% Functions

def show_train_history(data: tuple[list], metric: str, 
                       be: int=0, model_name: str='Unknown'):
    """
    Plots the metrics monitored during training.
    Parameters:
        data (tuple of lists): must be a tuple of size=2, where
         position 0 correspond to the metric evaluated on train set,
         and position 1 correspond to the metric evaluated on
         validation set.
        metric (str): the name of the metric contained in data. 
         Used for plotting.
        be (int): best epoch of the training of the model. Correspond
         to the best model returned after training.
        model_name (str): name of the model being tested. Used 
        for plotting.
    """
    train_metric = 0
    validation_metric = 1
    x_axis=np.arange(len(data[0]))+1
    
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, data[train_metric])
    plt.plot(x_axis, data[validation_metric])
    if be != 0: plt.axvline(x=be,color='r',ls='--')
    
    plt.title(f'Train History of {model_name}')
    plt.ylabel(metric.upper())
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.show()

def getMetrics(path: str, model_name: str):
    """
    Load to memory all metrics stored by saveModels.py
    Parameters:
        path (str): directory where the model data is stored.
        model_name (str): custom model's name to identify
         it inside directory path.
    Return:
        Returns tuple with all extracted information,
        position-wise:
            ( 0) Training accuracy
            ( 1) Training AUC
            ( 2) Training Loss
            ( 3) Best epoch
            ( 4) Time to train
            ( 5) Standard deviation of AUC during k-fold
            (extra) ...
            (-3) Confusion Matrix
            (-2) Overall metrics (pycm + sklearn)
            (-1) Class metrics (pycm + sklearn)
    """
    if path[-1] == '/': path = path[:-1]
    full_path = path + f"/Model_{model_name}/"
    
    # Load training evaluation
    path_temp = full_path + 'Training/'
    
    with open(path_temp + "ACC.pkl",'rb') as f:
        acc = pickle.load(f)
        acc_val = pickle.load(f)
    with open(path_temp + "AUC.pkl",'rb') as f:
        auc = pickle.load(f)
        auc_val = pickle.load(f)
    with open(path_temp + "LOSS.pkl",'rb') as f:
        loss = pickle.load(f)
        loss_val = pickle.load(f)
    with open(path_temp + "BestEpoch.pkl",'rb') as f:
        bestEpoch = pickle.load(f)
    with open(path_temp + "Time.pkl",'rb') as f:
        trainTime = pickle.load(f)
    try:
        with open(path_temp + "STD.pkl",'rb') as f:
            variation = {}
            try: variation['AUC'] = pickle.load(f)
            except: pass
            try: variation['ACC'] = pickle.load(f)
            except: pass
            try: variation['Spe'] = pickle.load(f)
            except: pass
            try: variation['Sen'] = pickle.load(f)
            except: pass
            try: variation['F1'] = pickle.load(f)
            except: pass
            try: variation['Pre'] = pickle.load(f)
            except: pass
    except: variation = {}
         
    # Load PYCM metrics
    path_temp = full_path + 'ConfusionMatrix/'
    
    with open(path_temp + "CM.pkl",'rb') as f:
        CM = pickle.load(f)
    with open(path_temp + "Overall.pkl",'rb') as f:
        Overall = pickle.load(f)
    with open(path_temp + "Class.pkl",'rb') as f:
        Class = pickle.load(f)
    
    return [acc,acc_val], [auc,auc_val], [loss,loss_val], bestEpoch, trainTime, variation, \
            CM, Overall, Class
    
def create_classWeight(support: dict):
    support = list(support.values())
    n_samples = sum(support)
    n_classes = len(support)
    wj = [0,0,0]
    for j in range(n_classes):
        wj[j] = n_samples/(n_classes*support[j])
    return wj
         
def selectMetrics(metrics: dict, select: list[str]):
    return [metrics[v] for v in select]

def renameMetrics(select: list[str]):
    # Original
    rename = {"AUC": "ROC-AUC",
              "Overall ACC": "Accuracy",
              "ACC": "Accuracy",
              "TPR": "Sensitivity",
              "TPR Macro": "Sensitivity",
              "TNR": "Specificity",
              "TNR Macro": "Specificity",
              "PPV": "Precision",
              "PPV Macro": "Precision",
              "F1": "F1-score",
              "F1 Macro":  "F1-score"
        }
    # For article
    rename = {"AUC": "AUC",
              "Overall ACC": "ACC",
              "ACC": "ACC",
              "TPR": "Recall",
              "TPR Macro": "Recall",
              "PPV Macro": "Precision",
              "F1": "F1",
              "F1 Macro":  "F1"
        }
    return [rename.get(v) or v for v in select]

def renameModels(names: list[str]) -> list:
    renamed_models = []
    for n in names:
        # Last minute changes:
        n = n.replace('_RED_','_REC_')   # because I changed the names
        n = n.replace('_REDA_','_AREC_') # because I changed the names
        n = n.replace('_C114_','_C115_') # because I had indexes wrong
        n = n.replace('_Inference','')   # because I dont want the large name
        
        if "sALSTM_Baseline_FC_v1_5fold" == n:
            n = "Baseline_5fold"
            
        if "BC_up_to_Gen11_5fold" == n:
            n = "GA LSTM-SA_5fold"
        
        if n[-4:] == 'fold':
            renamed_models.append(n[:-6])
        else:
            title = "Single train"
            # renamed_models = names.copy()
            # break
            renamed_models.append(n) # Fixed !!!
            
    else:
        title = f"Cross-Validation with {n[-5:]}"
    
    return renamed_models, title
    
def plotHistory(metrics: list, model_name: str, plot: bool=False):
    acc, auc, loss, bestEpoch = metrics
    if plot:
        show_train_history(acc,'acc', bestEpoch, model_name)
        show_train_history(auc,'auc', bestEpoch, model_name)
        show_train_history(loss,'loss', bestEpoch, model_name)
        
def plotTrain(path: str, metrics: list, model_name: str, be: np.array, plot: bool=False):
    """metrics: for compatibility to plotHistory()"""
    def _show_train_history(data, metric, ax, fn, be):
        """
        ax: object where to plot
        fn: fold number
        """
        train_metric = 0
        validation_metric = 1
        x_axis=np.arange(len(data[train_metric]))+1
        
        ax.plot(x_axis, data[train_metric], 'b',
                alpha=0.5, linewidth=2.5)
        ax.plot(x_axis, data[validation_metric], 'r',
                alpha=0.8, linewidth=2.5)
        if be != 0: ax.axvline(x=be,color='g',ls='--',
                               linewidth=1.8)
        
        ax.set_title(f'Train History (fold {fn})')
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'validation', 'best epoch'], loc='center right')
        ax.grid(True)

    if plot:
        try:
            path_name = f'{path}/Model_{model_name}/Training/TrainingCurves'
            model_name, _ = renameModels([model_name])
            model_name = model_name[0]
            
            with open(path_name, 'rb') as f:
                curves = pickle.load(f) # curves is a dict
        except FileNotFoundError:
            plotHistory(metrics, model_name, plot)
            return None
        # Default:
        n_graphs = 3
        size5 = (24,11)
        size2 = (16,11)
        size1 = (6,10)
        
        # Remove Accuracy #!!!
        curves.pop("ACC") 
        n_graphs = 2
        size5 = (24,8)
        size2 = (16,8)
        size1 = (6,8)
        # #################
        
        N = len(list(curves.values())[0])
        if N == 5: # (5 folds)
            fig, axs = plt.subplots(n_graphs,5, figsize=size5,sharex=True, dpi=300)
            fig.suptitle(model_name)
            for ax,(k,v) in zip(axs, curves.items()):
                # if k in ['acc', 'ACC']: continue # Don't show accuracy
                _show_train_history(v[0], k, ax=ax[0], fn=1, be=be if isinstance(be,int) else be[0])
                _show_train_history(v[1], k, ax=ax[1], fn=2, be=be if isinstance(be,int) else be[1])
                _show_train_history(v[2], k, ax=ax[2], fn=3, be=be if isinstance(be,int) else be[2])
                _show_train_history(v[3], k, ax=ax[3], fn=4, be=be if isinstance(be,int) else be[3])
                _show_train_history(v[4], k, ax=ax[4], fn=5, be=be if isinstance(be,int) else be[4])
        elif N == 2: # (2 folds)
            fig, axs = plt.subplots(n_graphs,2, figsize=size2,sharex=True, dpi=300)
            fig.suptitle(model_name)
            for ax,(k,v) in zip(axs, curves.items()):
                # if k in ['acc', 'ACC']: continue # Don't show accuracy
                _show_train_history(v[0], k, ax=ax[0], fn=1, be=be if isinstance(be,int) else be[0])
                _show_train_history(v[1], k, ax=ax[1], fn=2, be=be if isinstance(be,int) else be[1])
        elif N == 1: # (single fold)
            fig, axs = plt.subplots(n_graphs,1, figsize=size1,sharex=True, dpi=300)
            fig.suptitle(model_name)
            for ax,(k,v) in zip(axs, curves.items()):
                # if k in ['acc', 'ACC']: continue # Don't show accuracy
                _show_train_history(v[0], k, ax=ax, fn=1, be=be if isinstance(be,int) else be[0])   
        fig.tight_layout()
        plt.show()
        
        return curves
    else: return None
    
def plotOverallMetric_v1(metrics, Models, labels):
    """ Deprecated """
    # Bar plot
    df = pd.DataFrame(metrics, columns=labels)
    df_model = pd.DataFrame(Models,columns=['Model'])
    df = pd.concat([df_model,df],axis=1)
    
    df.plot(x='Model',
            figsize=(22,12),
            fontsize=18,
            rot=0,
            grid=True,
            kind='bar',
            ylim=[0.65, 1.00],
            title='Metrics per model')

    # df.to_csv("./EncoderDecoder/Models/ovr_metrics.csv")
    return df

def plotOverallMetric_v2(metrics: np.ndarray, Models: list, labels: list[str], 
                         filename:str='waterfall_plot.svg', plot=False):
    # Table
    labels = renameMetrics(labels)
    df = pd.DataFrame(metrics, columns=labels)
    df_model = pd.DataFrame(Models,columns=['Model'])
    df = pd.concat([df_model,df],axis=1)
    
    Models, title = renameModels(Models)
    
    def relative_to_prev(arr):
        metric = arr.copy()
        for i in range(len(metric)):
            base = arr[i-1]
            metric[i] = arr[i] if i==0 else arr[i]-base
        return metric#[::-1] # !!!
    
    def relative_to_first(arr):
        metric = arr.copy()
        base = arr[0]
        for i in range(len(metric)):
            metric[i] = base if i==0 else arr[i]-base
        return metric
    
    def roundNearestEven(n: int):
        return (n//2)*2
    
    if plot: # was nice: fig = go.Figure(layout=dict(height=1000,width=1800,margin=dict(l=20, r=20, t=15, b=30)))
        fig = go.Figure(layout=dict(height=600,width=1000,margin=dict(l=20, r=20, t=15, b=30)))
        M = len(Models)
        L = len(labels)
        factor=100
        
        relative_to = L*(['absolute']+(M-1)*['relative'])
        formated_labels = np.asarray([[f'<b>{l}</b>']*M for l in labels]).reshape(-1).tolist()
        formated_metrics = np.zeros(shape=metrics.T.shape)
        for i,m in enumerate(metrics.T):
            formated_metrics[i] = relative_to_prev(m) #!!!
        formated_metrics = formated_metrics.reshape(-1)*factor # flatten
        formated_text = []
        for i in range(len(formated_metrics)):
            m = relative_to[i]
            v = formated_metrics[i]
            if m == 'absolute':
                base = v.copy()
                formated_text.append(f'<b>{base:.1f}</b>')
            elif m == 'relative':
                m_prev = relative_to[i-1] == 'relative'
                v_prev = formated_metrics[i-1] if m_prev else 0
                formated_text.append(f'<b>{base + v_prev + v:.1f}</b>')
        

        for i in range(len(formated_metrics)):
            print(formated_metrics[i],'|',formated_labels[i],'|', relative_to[i])
            
        fig.add_trace(go.Waterfall(
            y = [formated_labels, L*Models],
            x = formated_metrics,
            # measure = L*(['absolute']+(M-1)*['relative']), # original
            measure = relative_to,
            orientation = "h",
            decreasing = {"marker":{"color":'#C04F15'}},
            increasing = {"marker":{"color":'#13501B'}},
            totals = {"marker":{"color":'#239133' }},
            textposition = "outside",
            text = formated_text,
            textfont=dict(size=20, color="black"), # CHANGED SIZE !!!
            width=0.85
        ))
            
        fig.update_layout(
            # title=title,
            waterfallgroupgap = 0.03,
            waterfallgap = 0.03,
            font_family="Times New Roman",
            font=dict(size=25,color="black"), # CHANGED SIZE !!!
            xaxis = dict(tickmode = 'linear',
                          tick0 = 0.70*factor,
                          dtick = 0.02*factor
                          ))
        xlim_min = 72
        # xlim_min = roundNearestEven(min(formated_metrics[::M])*0.98)
        fig.update_xaxes(title_font_family="Times New Roman",
                          range=[xlim_min, 0.95*factor],
                          showgrid=True, gridwidth=2.0, gridcolor='gray',
                          minor_griddash="dot", minor_gridwidth=1.5,
                          )
        fig.update_yaxes(title_font_family="Times New Roman",
                          showline=True, linewidth=1.5, linecolor='black')
        
        # Need kaleido: pip install -U kaleido
        # fig.write_image('Graphs_Plotly/tmp.png', scale=1)
        
        binary_data = \
            fig.to_image(format='svg',
                          width=None, height=None, 
                          scale = 50,
                          engine='orca'
                          )
        with open('Graphs_Plotly/' + filename, 'wb') as f: 
            f.write(binary_data)
            
        fig.show()

    return df
    
def plotCM(conf_mat: np.ndarray, 
           conf_mat_abs: np.ndarray, 
           model_name: str, which: bool=True, plot: bool=False):
    
    def underline(text):
        return "\u0332".join(text + ' ')[:-1]
    
    conf_mat = conf_mat if which else conf_mat_abs
    annot_format = '.4f' if which else '.0f'
    
    model_name, _ = renameModels([model_name])
    model_name = model_name[0]
    if plot:
        fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=128)
        ax.matshow(conf_mat, cmap=plt.cm.Oranges, alpha=0.75 if which else 0.85)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                # # Original
                # ax.text(x=j, y=i, s=f"{conf_mat[i, j]:{annot_format}}",
                #         va='center', ha='center', fontsize=21)
                # Combined
                ax.text(x=j, y=i, s=f"{conf_mat[i, j]:.4f}\n({conf_mat_abs[i, j]:.0f})",
                        va='center', ha='center', fontsize=21)
    
        plt.xlabel('Predictions', fontsize=22)
        plt.ylabel('Actuals', fontsize=22)
        plt.xticks(ticks=[0,1,2], labels=['Negative','Mixed','Positive'], 
                   fontsize=20)
        ax.xaxis.tick_bottom()
        plt.yticks(ticks=[0,1,2], labels=['Negative','Mixed','Positive'], 
                   rotation=90, fontsize=20,va='center')
        plt.title(f'{model_name}\n', fontsize=23)
        plt.show()
        
def plotCMpretty(conf_mat_abs: np.ndarray, 
                 model_name: str, plot: bool=False):
    
    conf_mat = conf_mat_abs
    
    model_name, _ = renameModels([model_name])
    model_name = model_name[0]
    if plot:
        # fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=128)
        # get pandas dataframe
        df_cm = pd.DataFrame(conf_mat, index=range(1, 4), columns=range(1, 4))
        # colormap: see this and choose your more dear
        cmap = 'Oranges'
        pp_matrix(df_cm, cmap=cmap,
                  pred_val_axis='x',
                  figsize=(10,10))
    
        # plt.xlabel('Predictions', fontsize=22)
        # plt.ylabel('Actuals', fontsize=22)
        # plt.xticks(ticks=[0,1,2], labels=['Negative','Mixed','Positive'], 
        #            fontsize=20)
        # ax.xaxis.tick_bottom()
        # plt.yticks(ticks=[0,1,2], labels=['Negative','Mixed','Positive'], 
        #            rotation=90, fontsize=20,va='center')
        # plt.title(f'{model_name}\n', fontsize=23)
        # plt.show()
    
def plotClassMetric(data: np.ndarray, Models: list, 
                    metric: str, plot: bool=False):
    Models, title = Models # this title thing is to handle the fold ID
    N = len(Models)
    if plot:
        values = list(np.asarray(data * 100).T.flat)
        to_plot = pd.DataFrame({'model_name':3*Models, # this 3 is for each sentiment
                                'Sentiment':N*['Negative'] + 
                                            N*['Mixed'] + 
                                            N*['Positive'],
                                metric:values,
                                }
                               )

        fig, ax = plt.subplots(figsize=(9, 4),dpi=180)
        
        sns.barplot(data=to_plot, x='model_name', y=metric,
                    hue='Sentiment', ax=ax, edgecolor="white")
        shifts = {'Negative':[-0.27,'#0099ff'],
                  'Mixed':[0.0,'#ffad33'],
                  'Positive':[0.27,'#00ff00'],
                  }
        rotations = [0, 0, 0, 25, 25, 25, 30]
        for sent in ['Negative','Mixed','Positive']:
            s,c = shifts[sent]
            ax.plot([i+s for i in range(N)], # fix for different that 3 models
                    to_plot.query("Sentiment == @sent")[metric], 
                    lw=1.5, marker='o',linestyle='dashed', color=c)
        # ax.set_title(f'Comparison of metric: {metric} ({title})')
        ax.set_ylabel(metric, fontsize=15)
        ax.set_xlabel(' ')
        ax.set_ylim(min(values)*0.95, 100)
        plt.xticks(rotation=rotations[N-1], fontsize=13)
        plt.yticks(fontsize=12)
        ax.set_axisbelow(True)
        plt.grid(which='major', axis='y', alpha=0.7,
                 color='k', linestyle='solid')
        plt.grid(which='minor', axis='y', alpha=0.7,
                 color='gray', linestyle='dashed')
        plt.minorticks_on()
        ax.grid(True, axis='y')
        plt.legend(fontsize=8.5,title="Sentiment", title_fontsize=10)
        # plt.setp(ax.lines[:-3], linewidth=2, color='white')
        
    return pd.DataFrame(data, columns=['Negative','Neutral','Positive'],
                        index=Models)

def plotClassMetric_v2(data: np.ndarray, Models: list, 
                       metric: str, plot: bool=False):
    Models, title = Models # this title thing is to handle the fold ID
    N = len(Models)
    if plot:
        values = list(data.T.flat)
        macros = data.mean(axis=-1).tolist() * 3
        values = (np.asarray(values) - np.asarray(macros)) * 100
        to_plot = pd.DataFrame({'model_name':3*Models, # this 3 is for each sentiment
                                'Sentiment':N*['Negative'] + 
                                            N*['Mixed'] + 
                                            N*['Positive'],
                                metric:values,
                                })
        
        fig, axs = plt.subplots(1,N+1,figsize=(10, 3.5),
                                dpi=200,sharey=True,
                                gridspec_kw={'width_ratios': N*[1] + [0.4]})
        macros = data.mean(axis=-1) * 100
        print(macros)
        for n in range(N):
            sns.barplot(data=to_plot.query('model_name == @Models[@n]'), 
                        x='model_name', y=metric,
                        hue='Sentiment', ax=axs[n], edgecolor="white",
                        palette=['#008ae6','#ffa31a','#00e600'])
            
            axs[n].plot([-0.5,0.5], [0.05,0.05],
                        linewidth=1.0, color='k')
            coords = (-0.53, 1.2) if N == 2 else (-0.53, 1.2)
            axs[n].annotate(f"{macros[n]:.1f}", coords,
                            bbox=dict(boxstyle="square", pad=0.12, fc="w"),
                            fontsize=12)
            axs[n].legend_.remove()
            axs[n].set_xlabel('')
            axs[n].set_xticklabels(['\n' + Models[n]], fontsize=14)
            
            axs[n].grid(which='major', axis='y', alpha=0.7,
                      color='k', linestyle='solid')
            axs[n].grid(which='minor', axis='y', alpha=0.7,
                      color='gray', linestyle='dashed')
            axs[n].minorticks_on()
            # axs[n].grid(True, axis='y')
            if n>0: axs[n].set_ylabel('', fontsize=1)
            else: 
                axs[n].set_ylabel(f'Δ{metric}', fontsize=14)
                axs[n].set_ylim(-20.0, 20.0)
            axs[n].set_yticklabels(axs[n].get_yticks(), fontsize=12)
            
            
        plt.axis('off')
        plt.plot([0], linewidth=6.0, color='#008ae6')
        plt.plot([0], linewidth=6.0, color='#ffa31a')
        plt.plot([0], linewidth=6.0, color='#00e600')
        plt.legend(['Negative','Neutral','Positive'],
                   fontsize=12,title="Sentiment", 
                   title_fontsize=14,
                   loc='center left')
        fig.tight_layout()
        plt.show()
    return pd.DataFrame(data, columns=['Negative','Neutral','Positive'],
                        index=Models)
        
def main(Models: list[str], metrics_path: str,                      # Set models
         filename: str,                                             # Set path to Waterfall 
         plot_Hist: bool=False,                                     # Plot learning curves
         plot_CM: bool=False, whichCM: bool=True,                   # Plot confusion matrix
         plot_Waterfall: bool=False, plot_Waterfallw: bool=False,   # Plot Waterfall of average metrics
         plot_Classes: bool=False, absoluteClasses: bool=False,     # Plot Barplots of class metrics
         selectOverall: list[str]=['AUC','F1 Macro','TNR Macro','TPR Macro','Overall ACC'],
         selectClass: list[str]=['AUC','F1','TNR','TPR','ACC']):
    
    hard_model = False
    info = ''
    metrics = {}
    plt_Overall = np.zeros((len(Models),len(selectOverall)))
    plt_wOverall = np.zeros((len(Models),len(selectOverall)))
    plt_Class   = np.zeros((len(Models),len(selectClass),3))
    std_Overall = []
    confMat = {}
    for i, model in enumerate(Models):
        info += f"\n[Model {model}]\n"
        # Plot training behaviour: #############################################################################################
        if model == "BaseModel [27]_5fold":
            current = metrics[model] = ([],[],[],[],[],[],
                                        {'Overall ACC': 0.777,'TPR Macro': 0.778,'PPV Macro': 0.780,
                                          'F1 Macro': 0.778,'AUC': 0.892},
                                        [])
            hard_model = True
        else:
            current = metrics[model] = getMetrics(metrics_path, model)
        
        if not hard_model:
            plotTrain(metrics_path, current[:4], model, current[3], plot_Hist)
            info += f"\tTraining time: {strftime('%H:%M:%S', gmtime(current[4]))}\n"
        
        # Load variables to plot Overall and Class metrics: (plots are later, but I need the variables here)
        if not hard_model:
            _overall = current[-2]
            _class = current[-1]
        else:
            _overall = current[-2]
        # _overall['AUC'] = np.mean(list(_class['AUC'].values())) # FIX AUC FOR EncDec Models #!!!
        
        # Define the variable with the standard deviations: ####################################################################
        if not hard_model:
            _std = current[5] # (dictionary)
            keys = {'AUC':'AUC','Overall ACC':'ACC','F1 Macro':'F1','TPR Macro':'Sen','TNR Macro':'Spe'}
            for k,v in keys.items():
                m = _overall[k]
                s = _std.get(v) or 0.0
                info += f"\tAverage {v:>3}: {m:.4f} ± {s:.4f}\n"
            std_Overall.append(list(_std.values()))
        
        # Define variable to plot Confusion Matrix: ############################################################################
        if not hard_model:
            _CM = current[-3]
            support = _class['P'] # Support, for weigthed average
            _CMabs = np.zeros(shape=(3,3),dtype='int')
            for j in range(3):
                actual = _CM[j]
                w = support[j]
                _CMabs[j] = np.asarray(actual*w).round(0).astype('int')
            confMat[model] = _CMabs
            plotCM(_CM, _CMabs, model, whichCM, plot_CM)
            # plotCMpretty(_CMabs, model, plot_CM)
        
        # Define variable to plot the overall metrics ##########################################################################
        print('before selectMetrics:')
        print('\noverall:', _overall)
        print('\nselec metrics:', selectOverall)
        current_Overall = selectMetrics(_overall,selectOverall) # dict to array
        plt_Overall[i] = np.asarray(current_Overall) # array into matrix
        
        hard_model = False
        continue #!!!
        
        # Define variable to plot the Class metrics and Weighted Overall metrics: ##############################################
        current_Class = [list(d.values()) for d in selectMetrics(_class,selectClass)] # dict to array
        plt_Class[i] = np.asarray(current_Class) # array into matrix
        # try:
        weights = list(support.values()) # Support, for weigthed average
        current_wOverall = np.average(current_Class,axis=1,weights=weights) # average
        plt_wOverall[i] = np.asarray(current_wOverall) # array into matrix
        # except Exception as e:
        #     print("current_Class = ",current_Class,sep='\n',end='\n')
        #     print("support = ",support,sep='\n',end='\n')
        #     raise e
    
    # Plot the overall metrics
    print("\nOverall:", plt_Overall)
    Overall_table = plotOverallMetric_v2(plt_Overall, Models, selectOverall, 
                                         filename=filename, plot=plot_Waterfall) # Plot for thesis 
    wOverall_table = plotOverallMetric_v2(plt_wOverall, Models, selectOverall, 
                                          filename=filename, plot=plot_Waterfallw) # Plot for thesis 
    
    # Plot metrics per class ###################################################################################################
    Class_tables = {}
    for m in range(len(selectClass)):
        if absoluteClasses:
            Class_tables[selectClass[m]] = \
                plotClassMetric(plt_Class[:,m,:], 
                                 renameModels(Models), 
                                 renameMetrics([selectClass[m]])[0],
                                 plot=plot_Classes)
        else:
            Class_tables[selectClass[m]] = \
            plotClassMetric_v2(plt_Class[:,m,:], 
                                renameModels(Models), 
                                renameMetrics([selectClass[m]])[0], 
                                plot=plot_Classes)
    
    # Modify the Overall_table to include the standard deviation, if any: ######################################################
    try:
        j=0
        std_Overall = np.asarray(std_Overall)
        for i in range(2,11):
            c = i-1
            if i%2 != 0: 
                continue
            Overall_table.insert(i, Overall_table.columns[c] + ' (σ)',
                                  std_Overall[:,j], allow_duplicates=True)
            wOverall_table.insert(i, wOverall_table.columns[c] + ' (σ)',
                                  std_Overall[:,j], allow_duplicates=True)
            j+=1
    except: 
        print("STD exception")
        pass
    # Print all information gathered to the console ############################################################################
    print(info)
    ############################################################################################################################
    return metrics, Overall_table, wOverall_table, Class_tables, confMat

#%% Define Functions for Post-Analysis

def createTableOfSelected(selected: list[int], metrics: pd.DataFrame):
    """
    This takes the selected combinations and creates an ordered table.
    Applies all formatting to directly create the table for thesis
    
    """
    def sort_function(values):
        # Considering all
        # sort_by = ['Accuracy','Avg. Sensitivity','Avg. Specificity','Avg. F1-score','AUC (ROC)']
        # coefs = np.array([0.20, 0.25, 0.10, 0.25, 0.20])
        
        # Considering less
        sort_by = renameMetrics(['ACC', 'TPR', 'F1','AUC'])
        coefs = np.array([0.10, 0.30, 0.40, 0.20]) # Formula in Thesis
        
        # # Considering also TNR
        # sort_by = renameMetrics(['ACC', 'TPR', 'TNR', 'F1','AUC'])
        # coefs = np.array([0.10, 0.30, 0.10, 0.40, 0.20]) # Formula in Thesis
        
        print(values[sort_by])
        values = values[sort_by].values
        return np.dot(values,coefs)
    
    def get_source(metric:str) -> list[str]:
        names = renameMetrics([metric])
        names.append(names[0] + ' (σ)')
        return names
    
    modelsComb = [int(mn.split('_')[-2][1:]) in selected for mn in metrics.Model]
    filtered = metrics[modelsComb].drop(['Model'],axis=1)
    scores = sort_function(filtered)
    
    
    table = pd.DataFrame(
        {'ID':filtered.index,
         'ACC (σ)':[f"{m*100:.1f} ({s*100:.2f})" for m,s in filtered[get_source('ACC')].values],
         'TPR (σ)':[f"{m*100:.1f} ({s*100:.2f})" for m,s in filtered[get_source('TPR')].values],
         'F1 (σ)':[f"{m*100:.1f} ({s*100:.2f})" for m,s in filtered[get_source('F1')].values],
         'AUC (σ)':[f"{m*100:.1f} ({s*100:.2f})" for m,s in filtered[get_source('AUC')].values],
         }
        )
    scores_index = scores.argsort()[::-1]
    print(scores_index)
    print(scores[scores_index])
    
    return table.iloc[scores_index]

def plotWhiskerSummary(metrics: pd.DataFrame, kind: str):
    """
    Uses the summary dataframe to create these whisker plots.
    This way I show all data, not only top 5.
    I can still maintain the table, but understanding the big picture.
    """
    is_REC = metrics.Model.apply(lambda x: 'AREC' if 'Att' in x else 'REC')
    metrics.insert(0,"Arch", is_REC)
    
    # Select metrics
    use = ['PPV','F1','TPR','ACC','TNR','AUC','Arch']
    use = renameMetrics(use)
    metrics = metrics[use]
    use = use[:-1]
    
    arch = metrics.pop('Arch')
    for i,m in enumerate(use):
        aux = metrics.pop(m)
        if i==0:
            formated_metrics = pd.DataFrame(dict(Values=aux,
                                                 Metrics=m,
                                                 Architecture=arch))
        else:
            formated_metrics = pd.concat(
                [formated_metrics,
                 pd.DataFrame(dict(Values=aux,
                                   Metrics=m,
                                   Architecture=arch))
                 ],ignore_index=True
                )
    fig, ax = plt.subplots(figsize=(8,6),dpi=200)   
    if kind == 'box':
        sns.boxplot(data=formated_metrics, ax=ax, orient='h',
                    x='Values', y='Metrics', hue='Architecture',
                    # palette=["#1f77b4", "#2ca02c"],
                    saturation=0.95, width=0.8, 
                    fliersize=5, linewidth=0.8, 
                    notch=False, showcaps=True,
                    flierprops=dict(marker="x"),
                    medianprops=dict(linewidth=1.2),
                    )
    elif kind == 'violin':
        sns.violinplot(data=formated_metrics, ax=ax, orient='h',
                       x='Values', y='Metrics', hue='Architecture',
                       inner=None,
                       # palette=["#1f77b4", "#2ca02c"],
                       saturation=0.95, width=1.0, split=True,
                       linewidth=1.0,
                       )
    elif kind == 'strip':
        sns.stripplot(data=formated_metrics, ax=ax, orient='h',
                       x='Values', y='Metrics', hue='Architecture',
                       # inner=None,
                       # palette=["#1f77b4", "#2ca02c"],
                       # saturation=0.95, 
                       size=1.0, 
                       split=True,
                       # linewidth=1.0,
                       )
    plt.grid(which='major', axis='x', 
             linewidth=0.9, color='k',
             linestyle='dashed', alpha=0.9)
    plt.grid(which='major', axis='y', 
             linewidth=0.2, color='gray',
             linestyle='solid', alpha=0.95)
    plt.grid(which='minor', axis='x', 
             linewidth=0.4, color='k',
             linestyle='dotted')
    plt.minorticks_on()
    ax.set_xlabel('')
    ax.set_ylabel('')
    return metrics
    
#%% MAIN

if __name__ == '__main__':
    # Define control variables: ===============================================
        
    metrics_path = './TrainedModels&Metrics'
    
    Models = [
              # "CSL_REDA_C127_5fold",
              # "sALSTM_Baseline_FC_v0_5fold",
              # "sALSTM_Baseline_FC_v1_5fold",
              "BaseModel [27]_5fold",
              "BC_up_to_Gen11_5fold",
              # "BC_Balanced_up_to_Gen11_5fold", # balanced with und and ove sampling !!!
              # "sALSTM_Baseline_FC_v1.1_5fold",
              "Waterfall_plot_Final_with_sampling.svg"
              ]   

    # =========================================================================
    
    if 'svg' in Models[-1]:
        OverallFilename = Models.pop(-1)
    else:
        OverallFilename = 'waterfall_plot.svg'
    
    selectOverall = ['TPR Macro','PPV Macro','Overall ACC','AUC',]
    selectClass = ['TPR','PPV','ACC','AUC',]
    
    # Run process to extract and plot metrics #!!!
    All_metrics, Overall_metrics, wOverall_metrics, \
        Class_metrics, confMat = main(Models,
                                      metrics_path, OverallFilename,
                                      plot_Hist=0, 
                                      plot_CM=0, whichCM=True, # True is normalized
                                      plot_Waterfall=1,
                                      plot_Waterfallw=0,
                                      plot_Classes=0, absoluteClasses=True, # False is variation
                                      selectOverall=selectOverall,
                                      selectClass=selectClass,
                                      )
    Overall_metrics_summary = Overall_metrics.describe().T
    wOverall_metrics_summary = wOverall_metrics.describe().T
    for k,v in Class_metrics.items():
        print(f"\n {k:>3} {60*'='}")
        print(v)
    
    # Only if 2 model are read (first is baseline and second is new)
    if len(Models) == 2: # Calculate ROI (relative overall improvement)
        def quick_renameModels(names: list[str]) -> list:
            return names, f"Cross-Validation with {names[0][-5:]}"
        
        (A, B), title = quick_renameModels(Models)
        metric = renameMetrics(['ACC','TPR','F1','AUC'])
        
        for m in metric:
            mA = Overall_metrics[Overall_metrics.Model == A][m].item()
            mB = Overall_metrics[Overall_metrics.Model == B][m].item()
            ROI = (mB-mA)/(1-mA)
            print(f"\nROI between {A} and {B}")
            print(f"based on {m} ({title}):")
            print(f"\t{mA*100:.2f}% --> {mB*100:.2f}%")
            print(f"\tROI: {ROI*100:+.2f}%")
    
    if len(Models) in [127, 128]: # Models from optimization: Tops Table
        n = 5
        # Find Top 5 models per metric
        topACC = Overall_metrics.sort_values(by=renameMetrics(['ACC']), ascending=False).head(n)
        topTPR = Overall_metrics.sort_values(by=renameMetrics(['TPR']), ascending=False).head(n)
        # topTNR = Overall_metrics.sort_values(by=renameMetrics(['TNR']), ascending=False).head(n)
        topF1S = Overall_metrics.sort_values(by=renameMetrics(['F1']), ascending=False).head(n)
        topAUC = Overall_metrics.sort_values(by=renameMetrics(['AUC']), ascending=False).head(n)
    
        # Merge the top 5 avoid considering the same model more than once
        topMerged = set()
        for top in [topACC,topTPR,topF1S,topAUC]:
            topMerged.update(top.index)    
        print(topMerged,'length =', len(topMerged))
        
        # Create table for Thesis 
        TableThesis = \
        createTableOfSelected(topMerged, metrics=Overall_metrics)
        
        TableAppendix = \
        createTableOfSelected(Overall_metrics.index, metrics=Overall_metrics)
        
        # Notes to myself
        # REC model (EncDec_BestChrom) need a +1 on the cleaning ID (C##+1)
        # In REC, C0 is one cleaning step (the first combination)
        # In REC, C126 is all the cleaning steps (the last combination)
        # In REC, C127 does not exist
        # After (+1) is done, C0 disappears, and C127 must be deleted
        
        # AREC model (EncDecAtt_BestChrom) does not need +1.
        # In AREC, C0 is no-cleaning step (is not a combination)
        # In AREC, C1 is one cleaning step (the first combination)
        # In AREC, C126 is the previous of all the cleaning steps
        # In AREC, C127 is all the cleaning steps (the last combination)
        # C0 and C127 must be deleted to be consistent with REC
        
    if len(Models) == 254: # Models from optimization: Whisker plots

        # _ = plotWhiskerSummary(Overall_metrics.copy(), 'box')
        _ = plotWhiskerSummary(Overall_metrics.copy(), 'violin')
        # _ = plotWhiskerSummary(Overall_metrics.copy(), 'strip')
            
#%% Exceptions on plots ...
##############################################################################
#############################################################################          
if False:

    def plotTrain(path: str, metrics: list, model_name: str, be: np.array, plot: bool=False):
        """metrics: for compatibility to plotHistory()"""
        def _show_train_history(data, metric, ax, fn, be):
            """
            ax: object where to plot
            fn: fold number
            """
            train_metric = 0
            validation_metric = 1
            x_axis=np.arange(len(data[train_metric]))+1
            
            ax.plot(x_axis, data[train_metric], 'b',
                    alpha=0.5, linewidth=3.5)
            ax.plot(x_axis, data[validation_metric], 'r',
                    alpha=0.8, linewidth=3.5)
            if be != 0: ax.axvline(x=be,color='g',ls='--',
                                   linewidth=2.5)
            ax.axvline(x=7,color='k',ls='--',
                                   linewidth=2.5)
            
            # ax.set_title(f'Train History (fold {fn})')
            if k == 'AUC':
                ax.set_yticks(np.linspace(0.90,1.00,num=6))
                ax.set_yticklabels([f"{n:.2f}" for n in np.linspace(0.90,1.00,num=6)],
                                   fontsize=18)
            elif k == 'LOSS':
                ax.set_yticks(np.linspace(0.0,1.0,num=5))
                ax.set_yticklabels([f"{n:.2f}" for n in np.linspace(0.0,1.0,num=5)],
                                   fontsize=18)
                
            
            
            ax.set_ylabel(metric.upper(), fontsize=18)
            ax.legend(['Train', 'Validation', 'Best epoch', 'Selected epoch'], 
                      loc='center right', fontsize=15)
            ax.grid(True)
            
            ax.grid(which='major', axis='y', alpha=0.7,
                     color='k', linestyle='solid')
            ax.grid(which='minor', axis='y', alpha=0.7,
                     color='gray', linestyle='dashed')
            ax.minorticks_on()
    
        if plot:
            try:
                path_name = f'{path}/Model_{model_name}/Training/TrainingCurves'
                model_name, _ = renameModels([model_name])
                model_name = model_name[0]
                
                with open(path_name, 'rb') as f:
                    curves = pickle.load(f) # curves is a dict
            except FileNotFoundError:
                plotHistory(metrics, model_name, plot)
                return None
        
            fig, axs = plt.subplots(2,1, figsize=(12,13),sharex=True, dpi=300)
            # fig.suptitle(model_name, fontsize=20)
            
            curves.pop("ACC")
            for ax,(k,v) in zip(axs, curves.items()):
                _show_train_history(v[0], k, ax=ax, fn=1, be=be if isinstance(be,int) else be[0])
            
            ax.set_xlabel('Epoch', fontsize=18)
            ax.set_xticks(list(range(0,36,5)))
            ax.set_xticklabels(list(range(0,36,5)), fontsize=17)
            
            
            
            # fig.tight_layout()
            plt.show()
            
            return curves
        else: return None
    
    plotTrain(metrics_path, 
              All_metrics['CSL_RED_BC_C114_Rpi'][:4], 
              'CSL_RED_BC_C114_Rpi', 
              All_metrics['CSL_RED_BC_C114_Rpi'][3], 
              True)



