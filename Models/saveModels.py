# -*- coding: utf-8 -*-
"""
Script to save all gathered metrics and info from model
"""
import os
import pickle
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide INFO messages


DefaultPath: str = "./Models"
MODEL: str = "ModelSave"
TRAINING: str = "Training"
CONFUSION_MAT: str = "ConfusionMatrix"


def save(model, model_name, model_format='', path=DefaultPath, **data):
    """
    This function save the `model` passed in the specified path, and
    also stores all metrics passed as *kargs.

    Parameters
        model: keras model to store
        model_name: (str) identifier of the model. Will be created a directory
        under `path` named `Model_<model_name>`
        
        model_format: (str) format to save the model. Default is '', which will
        save in format `savemodel`
        
        path: (str) directory where metrics are being stored. After this 
        function runs, will create 3 new directories under 
        `path/Model_<model_name>`:
            ModelSave: stores the keras model
            Training: stores metrics and info available during training. Metrics
            stored here are identified by keyword argument `train<name>`
            ConfusionMatrix: stores Pycm metrics. Values stores here are identified
            by keyword argument `pycm<name>`
            
        **kargs: (any) Parameters to be saved. `train<name>` must be passed as a list.
        Example: save(..., trainBestepoch=[best_epoch_value], ...)
    -------
    Returns: None.
    """  
    trainMetrics = []
    pycmMetrics = []
    save_vocab = False
    for key, value in data.items():
        if key[:5] == 'train':
            trainMetrics.append( (key,value) )
        elif key[:4] == 'pycm':
            pycmMetrics.append( (key,value) )
        elif key == 'vocab':
            save_vocab = True
            vocab_list = value
        else:
            print(f"Keyword argument `{key}` not identified")
            
    assert len(trainMetrics), "Train metrics are missing"
    assert len(pycmMetrics), "Pycm metrics are missing"
    log=''
    print("Saving metrics...")
    if path[:2] != './': path = './'+path
    if path[-1] == '/': path = path[:-1]
    model_path = create_dir(path,f"Model_{model_name}")
    
    # Save Training metrics:
    save_path = create_dir(model_path, TRAINING)
    for name,values in trainMetrics:
        name = name[5:]
        with open(save_path + f"/{name}.pkl", 'wb') as f:
            for n in range(len(values)):
                pickle.dump(values[n],f)
    log += f"\nTraining metrics correctly saved in {save_path}\n"
    
    # Save Pycm metrics:
    save_path = create_dir(model_path, CONFUSION_MAT)
    for name,values in pycmMetrics:
        name = name[4:]
        with open(save_path + f"/{name}.pkl", 'wb') as f:
            pickle.dump(values,f)
    log += f"\nPycm metrics correctly saved in {save_path}\n"
    if save_vocab:
        with open(model_path + "/vocab_list.pkl", 'wb') as f:
            pickle.dump(vocab_list,f)
    log += f"\nVocabulary correctly saved in {model_path + '/vocab_list.pkl'}\n"
    
    print(log)
    
def create_dir(path, new_dir):
    """
    path (str): path where directory want to be created.
     Must not end with '/' character!
    new_dir (str): name of new directory that will be 
     created inside path.
    """
    if new_dir not in os.listdir(path[2:]):
        os.mkdir(path +'/'+ new_dir)
    return path +'/'+ new_dir

#%% END

if __name__ == '__main__':
    # Run simple test
    import numpy as np
    
    # set dummy data to store
    model = Model([],[])
    model_name = "dummy"
    best_epoch = 3
    time_to_train = 1435.645
    acc = np.random.randn(5)
    auc = np.random.randn(5)
    loss = np.random.randn(5)
    global_values_to_store = {'Overall ACC': 0.70,'Kappa': 0.63,'TPR Macro': 0.82,'TNR Macro': 0.76,
                              'PPV Macro': 0.43,'F1 Macro':0.59,'Hamming Loss':0.23}
    class_values_to_store = {'ACC':{0: 0.85, 1: 0.53, 2: 0.72},'AUC':{0: 0.88, 1: 0.79, 2: 0.72}}
    cmTest = np.absolute(np.random.randn(3,3))
    vocab_list = ['mas', 'comida', 'restaurante']
    
    b=save([model],model_name,'',
         trainBestEpoch=[best_epoch],
         trainTime=[time_to_train],
         pycmOverall=global_values_to_store,
         pycmClass=class_values_to_store,
         trainACC=[acc,acc],
         trainAUC=[auc,auc],
         trainLOSS=[loss,loss],
         pycmCM = cmTest,
         vocab=vocab_list
         )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    