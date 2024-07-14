# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:30:39 2023
@author: Daniel Parada

This is the final implementation of the EncDec model, AFTER optimization.
The code is based on the BEST CHROMOSOME of genetic algorithm and have some
upgrades (may or may not), such as:
    -> Modify regularization techniques, including:
        L1/L2 regularization in RNN layers
        dropout percentage and position in network
        label smoothing, optimizer and learning rate
    -> Embedding arquitecture (normal or token+position)

This scripts allows the usage of the training of the model with a cross
validation pipeline (10-folds, for instance), or single training.

Cross validated model's metrics can be saved for future inspection (getMetrics)
Single trained model (keras Model + metrics) can be saved for future usage or
inspection (generalClassifier and getMetrics).

This script saves the model's results in a compatible way for loading with
class gC.Classifier3().
    
===============================================================================
===============================================================================

I won't modify this model anymore!
TO DO:
    - Run model with ImBalanced dataset
    - Run model with OVR sampled dataset
    - Run model with UND sampled dataset
    
    - Run 127 combinations of preprocessing steps (2 folds)
    - Run 5-fold cross-validation
    - Run with only TRAIN and TEST (merge train and validation) (maybe not)
    
    - Run cross-dataset validation (make another script)
    
Results:
    - Comparison of performances between different sampling methods
    - Determine best set of preprocessing steps
    - Find out average and stardard deviation of ACC, AUC, F1, Spe and Sen 
    
Table:
    
Model name                        Description                                             State
EncDecAtt_BestChrom_v2                Baseline (for usage, find best)                     waitlist
EncDecAtt_BestChrom_v2_5fold          Model with 5-fold cross-validation, to get metrics  waitlist
EncDecAtt_BestChrom_v2_C125_5fold     Model with 5-fold cross-validation, to get metrics  waitlist
EncDecAtt_BestChrom_v2_OVR_5fold      Model with Hybrid-sampled dataset                   waitlist
EncDecAtt_BestChrom_v2_OVR_C125_5fold Model with Hybrid-sampled dataset                   waitlist
EncDecAtt_BestChrom_v2_UND_5fold      Model with Under-sampled dataset                    waitlist
EncDecAtt_BestChrom_v2_UND_C125_5fold Model with Under-sampled dataset                    waitlist

EncDecAtt_BestChrom_v2_C0            Model with cleaning combination 0                    waitlist (no cleaning)
EncDecAtt_BestChrom_v2_C1            Model with cleaning combination 1                    waitlist
...                               ...                                                 ...
    
"""
#%% Parameters #!!!
frac: float      = 1.00   # for fast training or check dependency with dataSize
Vocab_Size: int  = 11560 
max_seq_len: int = 100
vector_size: int = 512
BATCH_SIZE: int  = 128
min_delta:float  = 0.0005 
patience: int    = 35
NN_epochs: int   = 150
verbose: int     = 1

#%% Control variables
comb = 101 # -1:default, 0:comb0, 1:comb1, ..., 99: comb99 (BEST)
NumberFolds = 5
NumberOverall = 53 # Pycm documentation (only int and float)
NumberClass   = 55 # Pycm documentation (only int and float)
save_fold_indexes = False # USE FOR 5-FOLD OR 10-FOLD, to select best
save = True
savePath = './TrainedModels&Metrics/THESIS'
# savePath = './EncoderDecoderAttention/Models'
# model_name = "EncDecAtt_BestChrom_v2" # old_name

# model_name = "CSL_REDA_BC" # Intermediate and Best cleaning
model_name = "CIL_REDA_BC" # Without cw

#%% Load libraries:
import os
import pickle
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide INFO messages
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time, strftime, gmtime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# Keras libraries:
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Bidirectional as Bi
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import TextVectorization
from keras_nlp.tokenizers import WordPieceTokenizer
from main.selfAttention import (TokenAndPositionEmbedding,
                                                build_pos_matrix,
                                                MHAttention)
# Libraries created localy:
from tools import merge_pos_neg_neutral
from Tokenizer import VocabBuilder
from Cleaning.textPreprocessing import clean
from Models import saveModels
from Models.getMetrics import show_train_history, selectMetrics
# Libraries for results:
from sklearn.metrics import classification_report, roc_auc_score
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
        
def create_classWeight(tr, n_classes=3):
    n_samples = len(tr)
    wj = {}
    for j in range(n_classes):
        wj[j] = n_samples/(n_classes*len(tr[tr==j]))
    print("class weights = ",wj)
    return wj

def build_model(XTrain, model_name):
    # Set hyperparameters
    VocabSize        = Vocab_Size #!!!
    NumberClasses     = 3
    NumberOfTimeSteps = 100
    Optimizer         = 'rmsprop'
    #                               
    Bidirectional     = ''
    NumberLSTMLayers  = 2
    PositionAttention = 'post'
    ModelDimension    = 512
    ShapeOfProjection = 128
    PercentageDropout = 0.1
    Decoder           = 'Pooling'
    ShapeOfDenseLayer = int(1.0*ModelDimension) 
    Activation        = 'tanh'
    Replicate         = 0
    Tokenizer         = 'WordPiece'
    if ModelDimension == 300: # Particular case
        DxH ={32: 30, 64: 60, 128: 100, 256: 150}
        ShapeOfProjection = DxH[ShapeOfProjection]
    NumberHeads       = int(ModelDimension/ShapeOfProjection)
    
    # Initialize Tokenizer
    if Tokenizer == "BlankSpace":
        _, VocabList, raw_embedding_matrix = \
            VocabBuilder.Word2VecVocabulary(XTrain, 
                                            vector_size=ModelDimension, 
                                            maxlen=NumberOfTimeSteps, 
                                            vocab_size=VocabSize)
        vocab_list = VocabList.copy()
        vocab_size = len(vocab_list)+2
        print("Vocab size:", vocab_size)
        
        vectorize_layer = TextVectorization(max_tokens=vocab_size,
                                            standardize=None,
                                            output_mode='int',
                                            output_sequence_length=NumberOfTimeSteps,
                                            vocabulary=vocab_list)
        #Fullfill vocabulary:
        for token in ['[UNK]','[PAD]']: # include new tokens
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,ModelDimension))
        UNKvector = np.zeros((1,ModelDimension))
        embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
        position_matrix = build_pos_matrix(NumberOfTimeSteps,ModelDimension)
        
    elif Tokenizer == "WordPiece":
        _, VocabList, raw_embedding_matrix = \
            VocabBuilder.WordPieceVocabulary(XTrain, 
                                             vector_size=ModelDimension, 
                                             maxlen=NumberOfTimeSteps, 
                                             vocab_size=VocabSize)
        vocab_list = VocabList.copy()
        vocab_size = len(vocab_list)+1
        print("Vocab size:", vocab_size)
        #Fullfill vocabulary:
        UNK_idx = 0
        for i,w in enumerate(vocab_list):
            if w == '[UNK]': UNK_idx = i
        _ = vocab_list.pop(UNK_idx)
        for token in ['[UNK]','[PAD]']: # include new tokens
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,ModelDimension))
        UNKvector = raw_embedding_matrix[UNK_idx:UNK_idx+1]
        raw_embedding_matrix = np.concatenate([raw_embedding_matrix[:UNK_idx,:],raw_embedding_matrix[UNK_idx+1:,:]])
        embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
        position_matrix = build_pos_matrix(NumberOfTimeSteps,ModelDimension)
        
        vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                             sequence_length=NumberOfTimeSteps)
    
    # Build network
    if PositionAttention in ['pre', 'pre-post']: # Use Token & Positional embedding
        EmbeddingLayer = TokenAndPositionEmbedding(vocab_size,ModelDimension,
                                                   maxlen=NumberOfTimeSteps,
                                                   weights_tokens=embedding_matrix,
                                                   weights_position=position_matrix,
                                                   mask_zero=True,
                                                   name='TokenPosEmbedding')
    elif PositionAttention in ['No','post']: # Use Token embedding
        EmbeddingLayer = Embedding(vocab_size,ModelDimension,
                                   input_length=NumberOfTimeSteps,
                                   weights=[embedding_matrix],
                                   mask_zero=True,
                                   name='TokenEmbedding')
            
    # Define Input
    model_input = layers.Input(shape=(NumberOfTimeSteps,), name='InputSequence')
    embed_output = EmbeddingLayer(model_input)
    encoder_input = layers.Dropout(PercentageDropout)(embed_output)
    mask_input = EmbeddingLayer.compute_mask(model_input)

    temp = encoder_input
    for i in range(Replicate+1):
        # Encoder: PreAttention
        if PositionAttention in ['pre', 'pre-post']:
            PreAttentionLayer = MHAttention(heads=NumberHeads, 
                                            dim_K=ShapeOfProjection, 
                                            dim_V=ShapeOfProjection, 
                                            d_model=ModelDimension,
                                            activation=Activation,
                                            name=f'PreAttention{i+1}')
            attention_output = PreAttentionLayer(temp,temp,
                                                 attention_mask=mask_input)
        else:
            attention_output = temp
            
        # Recursive Encoder ==============================================
        ShapeOfLSTM = ModelDimension // 2 if Bidirectional == 'Bi' else ModelDimension
        RecursiveEncoder = f"{Bidirectional}(LSTM({ShapeOfLSTM}, return_sequences=True))"
        if NumberLSTMLayers == 1:
            encoder_output = eval(RecursiveEncoder)(attention_output)
        else:
            x = eval(RecursiveEncoder)(attention_output)
            for _ in range(NumberLSTMLayers-2):
                x = eval(RecursiveEncoder)(x)  
            encoder_output = eval(RecursiveEncoder)(x)
            
        if PositionAttention in ['post','pre-post']: 
            # Create Position embedding before postAttention layer
            encoder_output = encoder_output + position_matrix
            PostAttentionLayer = MHAttention(heads=NumberHeads, 
                                             dim_K=ShapeOfProjection, 
                                             dim_V=ShapeOfProjection, 
                                             d_model=ModelDimension,
                                             activation=Activation,
                                             name=f'PostAttention{i+1}')
            temp = PostAttentionLayer(encoder_output,encoder_output,
                                      attention_mask=mask_input)
        else:
            temp = encoder_output
        if PositionAttention == 'pre-post': break
    decoder_input = temp

    # Define Decoder ==============================================
    if Decoder == 'Dense':
        ContextDecoder = \
        Sequential([layers.Input((NumberOfTimeSteps,ModelDimension)),
                    layers.Flatten(),
                    layers.Dense(ModelDimension, activation=None),
                    layers.Activation(Activation)
                    ], name='Decoder_MLP')
    elif Decoder == 'Pooling':
        ContextDecoder = \
        Sequential([layers.Input((NumberOfTimeSteps,ModelDimension)),
                    layers.AveragePooling1D(pool_size=10,strides=8),
                    layers.GlobalAveragePooling1D()
                    ], name='Decoder_AvgPool')  
    elif Decoder == 'LSTM':
        ContextDecoder = LSTM(ModelDimension, return_sequences=False,
                              name='Decoder_LSTM')
    elif Decoder == 'Bi-LSTM':
        ContextDecoder = Bi(LSTM(ModelDimension//2, return_sequences=False,
                                 name='Decoder_BiLSTM'))
        
    decoder_output = ContextDecoder(decoder_input)

    # Define FFNN =================================================
    if ShapeOfDenseLayer != 0:
        ffnn_output = layers.Dense(ShapeOfDenseLayer, activation=None,
                                   name='FFNN')(decoder_output)
        ffnn_output = layers.Activation(Activation)(ffnn_output)
    else:
        ffnn_output = decoder_output
    ffnn_output = layers.Dropout(PercentageDropout)(ffnn_output)

    # Define classifier
    model_output = layers.Dense(NumberClasses, activation=None,
                                name='Classifier')(ffnn_output)
    model_output = layers.Activation('softmax')(model_output)

    model = Model(inputs=model_input, outputs=model_output)
    model.summary()

    model.compile(optimizer=Optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'AUC'])
    return model, vectorize_layer, vocab_list

#%% Load data: #!!!
df = pd.read_csv('./Data/ZomatoFinal/FULL_PT_100_ImBalanced2.csv',
                  usecols=['raw_text','text', 'rating'])
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df = merge_pos_neg_neutral(df, plot=False)
    
if frac < 1.0: df = df.sample(frac=frac,ignore_index=True)

# Dictionary to map labels to integers:
map_labels = {r:i for i,r in enumerate(np.unique(df['rating']))}
inv_map_labels = {v:k for k,v in map_labels.items()}
#__________

#%% Combination cycle
comb_set = range(-1,0) if comb<0 else range(comb,comb+1)
raw_model_name = model_name[:]
for c in comb_set:
    print(f"\nRunning combination {c}")
    if c >= 0: model_name = raw_model_name + f'_C{c}'
    print(50*"*",f"Model name: {model_name}", 50*"*", sep='\n')
    #%% Text cleaning
    if c < 0:
        print("DEFAULT Text cleaning.")
    else:
        cln = Cleaning()
        steps = ['accents','lowercase','emojis',
                  'stopwords','bigrams','punctuation','numbers']
        combinations = []
        for i in range(1,8):
            temp = list(itertools.combinations(steps,i))
            combinations.extend(temp)
        combinations.insert(0,[]) # Include No-cleaning procedure
        steps = list(combinations[c])
        cln.define_steps(steps)
        # Perform cleaning
        print("Text cleaning", steps)
        df['text'] = cln.Clean(df['raw_text'],True)
        # Remove empty reviews
        df['text'].replace('', np.nan, inplace=True)
        df.dropna(subset=['text'], inplace=True)
        df.reset_index(drop=True,inplace=True)
    
    #%% Train model (CrossValidation or Single run)
    kf = StratifiedKFold(n_splits=NumberFolds, shuffle=True)
    data_groups = kf.split(df, df.rating)
        
    foldOverall = np.zeros((NumberFolds, NumberOverall+1)) # +1 to include AUC
    foldClass   = np.zeros((NumberFolds, NumberClass, 3))
    foldAUC = np.zeros(NumberFolds)
    foldACC = np.zeros(NumberFolds)
    foldSpe = np.zeros(NumberFolds)
    foldSen = np.zeros(NumberFolds)
    foldF1  = np.zeros(NumberFolds)
    foldCM = np.zeros((NumberFolds, 3, 3))
    foldTime = np.zeros(NumberFolds)
    bestEpoch = np.zeros(NumberFolds)
    
    historyACC  = []
    historyAUC  = []
    historyLOSS = []
    
    log = ['']*NumberFolds
    data_folds = [0]*NumberFolds
    for fn, (tr,te) in enumerate(data_groups):
        if False: # For testing
            fn = 0
            (tr,te) = next(data_groups)
            
        print("fold:", fn)
        data_folds[fn] = (tr,te)
        
        # Extract features and labels:
        if NumberFolds == 5:
            df_train = df.iloc[tr]
            df_train, df_val = train_test_split(df_train, test_size=0.25,
                                                shuffle=True, stratify=df_train['rating'])
            df_test   = df.iloc[te]
        elif NumberFolds == 2:
            df_train = df.iloc[tr]
            df_test   = df.iloc[te]
            df_test, df_val = train_test_split(df_test, test_size=0.40,
                                               shuffle=True, stratify=df_test['rating'])
            
        trainSize = len(df_train)
        validSize = len(df_val)
        testSize  = len(df_test)
        log[fn] += f"Data size used: TRAIN {trainSize}, TEST {testSize}, VALIDATION {validSize}\n"
        datasetSize = trainSize + validSize + testSize
        trainSize /= datasetSize
        validSize /= datasetSize
        testSize  /= datasetSize
        print('Data Partitions:'
              f'TRAIN {trainSize*100:.2f}',
              f'VALID {validSize*100:.2f}',
              f'TEST {testSize*100:.2f}',) 
        
        # Extract features:
        train_features = df_train['text']
        valid_features = df_val['text']
        test_features  = df_test['text']
        
        # Extract labels:
        train_labels = np.array(list(map(lambda x: map_labels[x],df_train['rating'])))
        valid_labels = np.array(list(map(lambda x: map_labels[x],df_val['rating'])))
        test_labels  = np.array(list(map(lambda x: map_labels[x],df_test['rating'])))
        
        # Set CSL using class_weight parameter:
        cw = create_classWeight(train_labels) if 'CSL_' in model_name else None
        
        # One-hot encoding on labels:
        train_labels = tf.keras.utils.to_categorical(train_labels)
        valid_labels = tf.keras.utils.to_categorical(valid_labels)
        test_labels  = tf.keras.utils.to_categorical(test_labels)
        
        model, vectorize_layer, vocab_list = build_model(train_features, model_name)
        
        print("Tokenization layer...")
        t = time()
        trainData = vectorize_layer(train_features)
        validData = vectorize_layer(valid_features)
        testData = vectorize_layer(test_features)
        print(f"(Elapsed time: {strftime('%M:%S', gmtime(time()-t))})")
        
        # Define Callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode='max',
                                              patience=patience, min_delta=min_delta,
                                              verbose=1, restore_best_weights=True)
        def schedule(epoch, lr):
            if epoch < 3: return lr # warm up
            else:
                if lr < 1e-5: return lr # limiter
                exponential = lr * tf.math.exp(-0.1)
                return exponential
        sch = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    
        print("Training started...")
        t = time()
        history = model.fit(trainData,train_labels,
                            validation_data=(validData,valid_labels),
                            epochs=NN_epochs,
                            callbacks=[es,sch],
                            batch_size=BATCH_SIZE,
                            class_weight=cw,
                            verbose=0)
        foldTime[fn] = time()-t
        bestEpoch[fn] = es.best_epoch+1
        print("Training finished...")
        print(f"Training time: {strftime('%H:%M:%S', gmtime(foldTime[fn]))}")
    
        # Plot training:
        historyACC.append((history.history['accuracy'],history.history['val_accuracy']))
        historyAUC.append((history.history['auc'],history.history['val_auc']))
        historyLOSS.append((history.history['loss'],history.history['val_loss']))
        
        show_train_history(historyACC[-1],'acc', bestEpoch[fn], model_name)
        show_train_history(historyAUC[-1],'auc', bestEpoch[fn], model_name)
        show_train_history(historyLOSS[-1],'loss', bestEpoch[fn], model_name)
    
        # Results (Test Dataset):
        print("\nTesting model...")
        test_real_labels = np.argmax(test_labels, axis=-1)
        
        test_forecasts = model.predict(testData)
        test_y_pred = np.argmax(test_forecasts, axis=-1)
        
        classTest = classification_report(test_real_labels, test_y_pred)
        cmTest = ConfusionMatrix(test_real_labels, test_y_pred,is_imbalanced=True)
        ovr_AUC = roc_auc_score(test_labels, test_forecasts, average='macro',multi_class='ovr')
        foldAUC[fn] = ovr_AUC
        foldACC[fn] = cmTest.overall_stat['Overall ACC']
        foldSpe[fn] = cmTest.overall_stat['TNR Macro']
        foldSen[fn] = cmTest.overall_stat['TPR Macro']
        foldF1[fn] = cmTest.overall_stat['F1 Macro']
        
        cmTest.plot(cmap=plt.cm.Blues,
                    number_label=True,
                    normalized=True,    # Using normalized because dataset in imbalanced
                    plot_lib="seaborn",
                    title="TEST CONFUSION MATRIX")
        foldCM[fn] = cmTest.to_array(normalized=True)
        
        log[fn] += f"Training time: {strftime('%H:%M:%S', gmtime(foldTime[fn]))}\n"
        log[fn] += f"Best epoch: {bestEpoch[fn]}\n"
        log[fn] += f"Classification report TEST\n{classTest}\n"
        
        # Store metrics (assuming python dictionary is ordered)
        LabelsOverall = [k for k,v in cmTest.overall_stat.items() if isinstance(v,(float,int)) ]
        LabelsClass   = [k for k,v in cmTest.class_stat.items() if isinstance(v[0],(float,int))]
        
        foldOverall[fn] = selectMetrics(cmTest.overall_stat,LabelsOverall) +[ovr_AUC]
        foldClass[fn]   = np.asarray([list(d.values()) for d in selectMetrics(cmTest.class_stat,LabelsClass)])
        AUC_id = 36
        foldClass[fn,AUC_id] = roc_auc_score(test_labels, test_forecasts, average=None, multi_class='ovr') # use sklearn AUC
                
        # restores session after creating new model
        tf.keras.backend.clear_session()
    
    #%% Print Log and write to file:
    for f in range(NumberFolds):
        print(f"Log of fold #{f+1}:",log[f],sep='\n')
        with open(f'{savePath}/log_fold{f}.txt','w') as log_file:
            log_file.write(log[f])
    if save_fold_indexes:
        with open(savePath + "/data_folds.pkl",'wb') as f:
            pickle.dump(data_folds,f)
    
    #%% Save model
    if save:   
        FinalBestEpoch = bestEpoch # now it is an array
        FinaltrainTime = np.mean(foldTime)
        FinalVariation = [np.std(foldAUC),np.std(foldACC),np.std(foldSpe),np.std(foldSen),np.std(foldF1)]
        FinalOverall = {k:v for k,v in zip( LabelsOverall+['AUC'], np.mean(foldOverall,axis=0) )}
        FinalClass = {k:{j:v for j,v in enumerate(varray)} for k,varray in zip( LabelsClass, np.mean(foldClass,axis=0) )}
        FinalACC = historyACC[-1]
        FinalAUC = historyAUC[-1]
        FinalLOSS = historyLOSS[-1]
        FinalCM = np.mean(foldCM, axis=0)
        FinalVocab = vocab_list[2:] # vocab without [PAD] and [UNK]
        
        model = Model([],[]) # DUMMY MODEL
        model_name += f"_{NumberFolds}fold" # AVOID OVERWRITE
        print("Model name to save:", model_name)
    
        saveModels.save([model], model_name, '', path=savePath,
                        trainBestEpoch = [FinalBestEpoch],
                        trainTime = [FinaltrainTime],
                        trainSTD = FinalVariation,
                        pycmOverall = FinalOverall,
                        pycmClass = FinalClass,
                        trainACC = FinalACC, # keep for compatibility
                        trainAUC = FinalAUC, # keep for compatibility
                        trainLOSS = FinalLOSS, # keep for compatibility
                        pycmCM = FinalCM,
                        vocab = FinalVocab
                        )
        
        with open(savePath + f'/Model_{model_name}/Training/TrainingCurves', 'wb') as f:
            # Saves training curves as dict
            pickle.dump({'ACC':historyACC,
                         'AUC':historyAUC,
                         'LOSS':historyLOSS}, f)
