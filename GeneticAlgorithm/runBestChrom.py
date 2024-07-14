# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:30:26 2024

@author: Daniel

Run best chromosome:
    First chromosome from the last generation
    It must have self-Attention mechanisms
    Must not be large
"""
#%% Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide INFO messages

import json 
import pickle
from time import time, strftime, gmtime, localtime
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import np_utils
from pycm import ConfusionMatrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional as Bi
from tensorflow.keras.layers import TextVectorization
from keras_nlp.tokenizers import WordPieceTokenizer
# from Models.getMetrics import show_train_history
from main.selfAttention import (TokenAndPositionEmbedding,
                                                build_pos_matrix,
                                                MHAttention)
from tools import merge_pos_neg_neutral
from Models import saveModels
from Models.getMetrics import show_train_history, selectMetrics
from Cleaning.textPreprocessing import clean, FC
from Tokenizer import VocabBuilder
from GeneticAlgorithm.encoding import decode_chromosome

#%% Util functions

def json_to_tokens():
    """parse json file to list of list (not padded)"""
    with open("Corpus/tokenized_data.json", 'r', encoding='utf-8') as f:
        tokenized_data = json.load(f)
        
    review_ids = list(map(int,tokenized_data.keys()))
    tokens = list(tokenized_data.values())
        
    return review_ids, tokens

def create_df(review_ids, texts, ratings):
    return pd.concat([ratings, texts], axis=1, 
                     ignore_index=True).rename({0:'review_id',1:'rating',2:'text'},
                                               axis=1)
                                               
def create_classWeight(tr, n_classes=3):
    n_samples = len(tr)
    wj = {}
    for j in range(n_classes):
        wj[j] = n_samples/(n_classes*len(tr[tr==j]))
    print("class weights = ",wj)
    return wj

def schedule(epoch, lr):
    if epoch < 3: return lr # warm up
    if lr < 1e-5: return lr # limiter
    else: return lr * tf.math.exp(-0.1) # decay   

def _exception_msg(msg: str):
    print(30*'*', 
          msg, '', 
          30*'*', sep='\n')        

def classification_report(y, y_hat):
    pass                    

def under_sample(df, full_balance):
    labels = np.unique(df.rating)
    
    # Find the lowest class lenght:
    max_len = []
    for i in labels:
        lenght = len(df[df.rating == i])
        print("current max_len", max_len)
        print("len of label", i, "is", lenght)
        max_len.append(lenght)
    print(max_len)
    max_len = max_len[1]
    print(max_len)
    
    # Create new balanced dataset:
    df1 = pd.DataFrame(columns=df.columns)
    for i in labels:
        new_rows = df[df.rating == i].sample(max_len, replace=full_balance)
        df1 = pd.concat([df1, new_rows], ignore_index=False)
        
    percent = 100 - round((len(df)-len(df1))/len(df)*100,1)
    print(f"Using {round(percent,1)}% of dataset")
    
    return df1.sort_index()
                                               
#%% Load data
# *****************************************************************************
dataPath: str    = './Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv'
review_ids, texts = json_to_tokens()
texts = pd.Series(list(map(lambda x: clean(' '.join(x), *FC), texts)))
review_ids = pd.Series(review_ids)
ratings = merge_pos_neg_neutral(pd.read_csv(dataPath, usecols=['review_id','rating']))

DATA: pd.DataFrame = create_df(review_ids, texts, ratings)
# *****************************************************************************

#%% Parameters #!!!
NumberFolds:     int = 5
NumberOverall: int = 53 # Pycm documentation (only int and float)
NumberClass: int   = 55 # Pycm documentation (only int and float)
PatienteceValue: int = 37
EpochsValue:     int = 150
min_delta:     float = 0.0005
# Verbose
VerboseData:   int = 1
VerboseTrain:  int = 1

NUMBER_CLASSES:   int   = 3      # number of classes to consider for classification
NUMBER_TIMESTEPS: int   = 100    # number of tokens to consider for model input
BATCH_SIZE:       int   = 128    # Batch size on GPU
VOCAB_SIZE:       int   = 11_560 # vocabulary size of model input
LEARNING_RATE:    float = 0.0001  # learning rate of RMSProp optimizer

MetricsPath:     str = "TrainedModels&Metrics/"


SAMPLING_METHOD: int = 1
# Values: 
#    0 -> Normal
#    1 -> Undersampling
#    2 -> Oversampling

#%% Constructor

def chromossome_to_architecture(chromossome: list, TrainData: pd.Series):
    # Depend on the Individual  
    if False: # For Testing
        chromossome = ['Bi', 2, 'post', 512, 64, 0.1, 'LSTM', 0.5, 'relu', 0, 'WordPiece']
        TrainData = DATA['text'][:268346].copy()
    
    try:                       
        Bidirectional     =     chromossome[ 0] 
        NumberLSTMLayers  =     chromossome[ 1]
        PositionAttention =     chromossome[ 2]
        ModelDimension    =     chromossome[ 3] 
        ShapeOfProjection =     chromossome[ 4]
        PercentageDropout =     chromossome[ 5]
        Decoder           =     chromossome[ 6]
        ShapeOfDenseLayer = int(chromossome[ 7]*ModelDimension) 
        Activation        =     chromossome[ 8]
        Replicate         =     chromossome[ 9]
        Tokenizer         =     chromossome[10]
        
        if ModelDimension == 300: # Particular case
            DxH ={32: 30, 64: 60, 128: 100, 256: 150}
            ShapeOfProjection = DxH[ShapeOfProjection]
        NumberHeads       = int(ModelDimension/ShapeOfProjection)
        
        if PositionAttention == 'pre-post': # Particular case
            Replicate = 0
    
    except Exception as exc:
        _exception_msg("Error extracting Phenotypes")
        raise exc
    
    # Build Vocabulary and Tokenization Layer
    #=========================================
    if Tokenizer == "BlankSpace":
        _, VocabList, raw_embedding_matrix = \
            VocabBuilder.Word2VecVocabulary(TrainData, 
                                            vector_size=ModelDimension, 
                                            maxlen=NUMBER_TIMESTEPS, 
                                            vocab_size=VOCAB_SIZE)
        vocab_list = VocabList.copy()
        vocab_size = len(vocab_list)+2
        print("Vocabulary size:", vocab_size)
        
        vectorize_layer = TextVectorization(max_tokens=vocab_size,
                                            standardize=None, output_mode='int',
                                            output_sequence_length=NUMBER_TIMESTEPS,
                                            vocabulary=vocab_list)
        #Fullfill vocabulary:
        for token in ['[UNK]','[PAD]']: # include new tokens
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,ModelDimension))
        UNKvector = np.zeros((1,ModelDimension))
        embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
        position_matrix = build_pos_matrix(NUMBER_TIMESTEPS,ModelDimension)
        
    elif Tokenizer == "WordPiece":
        _, VocabList, raw_embedding_matrix = \
            VocabBuilder.WordPieceVocabulary(TrainData, 
                                             vector_size=ModelDimension, 
                                             maxlen=NUMBER_TIMESTEPS, 
                                             vocab_size=VOCAB_SIZE)
        vocab_list = VocabList.copy()
        vocab_size = len(vocab_list)+2
        print("Vocabulary size:", vocab_size)
        
        #Fullfill vocabulary:
        for token in ['[UNK]','[PAD]']: # include new tokens
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,ModelDimension))
        UNKvector = np.zeros((1,ModelDimension))
        embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
        position_matrix = build_pos_matrix(NUMBER_TIMESTEPS, ModelDimension)
        print("Size of Token Embedding matrix:", embedding_matrix.shape)
        print("Size of Position Embedding matrix:", position_matrix.shape)
        
        vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                             sequence_length=NUMBER_TIMESTEPS)
        
    try:
        # EMBEDDING LAYER ****************************************************************
        EmbeddingLayer = TokenAndPositionEmbedding(vocab_size,ModelDimension,
                                                   maxlen=NUMBER_TIMESTEPS,
                                                   weights_tokens=embedding_matrix,
                                                   weights_position=position_matrix,
                                                   mask_zero=True,
                                                   name='TokenPosEmbedding')
        # ******************************************************************************
        
        # MODEL INPUT ***************************************************************
        model_input = layers.Input(shape=(NUMBER_TIMESTEPS,), 
                                   # batch_size=BATCH_SIZE,
                                   name='Input_sequence')
        mask_input = EmbeddingLayer.compute_mask(model_input)
        embed_output = EmbeddingLayer(model_input)
        embed_output = layers.Dropout(PercentageDropout, 
                                      name='Input_dropout')(embed_output)
        # *****************************************************************
        
        temp = embed_output
        normalizationLayer = layers.Layer() # dummy
        for i in range(Replicate+1):
            # ATTENTION BEFORE *************************************************
            if PositionAttention in ['pre', 'pre-post']:
                PreAttentionLayer = MHAttention(heads=NumberHeads, 
                                                dim_K=ShapeOfProjection, 
                                                dim_V=ShapeOfProjection, 
                                                d_model=ModelDimension,
                                                activation='tanh',
                                                name=f'PreAttention{i+1}')
                attention_output = PreAttentionLayer(temp,temp,
                                                     attention_mask=mask_input)
                if i > 0:
                    attention_output = normalizationLayer(attention_output + temp)
                
            else:
                attention_output = temp
            # ******************************************************************
                
            # RECURRENT ENCODER ***************************************************************
            ShapeOfLSTM = ModelDimension // 2 if Bidirectional == 'Bi' else ModelDimension
            RecursiveEncoder = \
            f"""{Bidirectional}(LSTM({ShapeOfLSTM}, return_sequences=True,\
            dropout={PercentageDropout}, kernel_regularizer=regularizers.l2(0.0001)))"""
                                               
            if NumberLSTMLayers == 1:
                encoder_output = eval(RecursiveEncoder)(attention_output)
            else:
                x = eval(RecursiveEncoder)(attention_output)
                for k in range(NumberLSTMLayers-2):
                    x = eval(RecursiveEncoder)(x)  
                encoder_output = eval(RecursiveEncoder)(x)
            # *********************************************************
            
            if i > 0:
                encoder_output = normalizationLayer(attention_output + encoder_output)
            
            # ATTENTION AFTER *****************************************
            if PositionAttention in ['post','pre-post']: 
                # Create Position embedding before postAttention layer
                # encoder_output = encoder_output + position_matrix
                
                PostAttentionLayer = MHAttention(heads=NumberHeads, 
                                                 dim_K=ShapeOfProjection, 
                                                 dim_V=ShapeOfProjection, 
                                                 d_model=ModelDimension,
                                                 activation='tanh',
                                                 name=f'PostAttention{i+1}')
                temp = PostAttentionLayer(encoder_output,encoder_output,
                                          attention_mask=mask_input)
                if i > 0:
                    temp = normalizationLayer(temp + encoder_output)
            else:
                temp = encoder_output
            # *********************************************************
            
            # NORMALIZATION LAYER *****************************************
            if i == 0: # In the first iteration it does NOT Normalize it
                normalizationLayer = tf.keras.layers.Normalization(mean=0.0,
                                                                   variance=2.71828)
                # normalizationLayer.adapt(temp)
            # *********************************************************
            
        decoder_input = temp
        # DECODER ************************************************************
        if Decoder == 'Dense':
            ContextDecoder = \
            Sequential([layers.Input((NUMBER_TIMESTEPS,ModelDimension)),
                        layers.Flatten(),
                        layers.Dense(ModelDimension, activation=None),
                        layers.Activation(Activation)
                        ], name='Decoder_MLP')
        elif Decoder == 'Pooling':
            ContextDecoder = \
            Sequential([layers.Input((NUMBER_TIMESTEPS,ModelDimension)),
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
        # *********************************************************************
        
        # FEED FORWARD NETWORK *********************************************
        if ShapeOfDenseLayer != 0:
            ffnn_output = layers.Dense(ShapeOfDenseLayer, activation=None,
                                       name='FFNN')(decoder_output)
            ffnn_output = layers.Activation(Activation, 
                                            name='FFNN_act')(ffnn_output)
        else:
            ffnn_output = decoder_output
        ffnn_output = layers.Dropout(PercentageDropout, 
                                     name='FFNN_dropout')(ffnn_output)
        # *************************************************************
        
        # OUTPUT LAYER ************************************************
        model_output = layers.Dense(NUMBER_CLASSES, activation=None,
                                    name='Classifier')(ffnn_output)
        model_output = layers.Activation('softmax', 
                                         name='Classifier_act')(model_output)
        # *************************************************************
        
        model = Model(inputs=model_input, outputs=model_output)
        model.summary()
        
    except Exception as exc:
        _exception_msg("Error building the Model Architecture")
        with open(MetricsPath + 'logErrors.txt','a') as f:
            header = "[log at {}]: Error building the model: {}\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), exc)
            chrom_info = \
            [header,
            f"Bidirectional = {Bidirectional}\n",
            f"NumberLSTMLayers = {NumberLSTMLayers}\n",
            f"PositionAttention = {PositionAttention}\n",
            f"ModelDimension = {ModelDimension}\n",
            f"ShapeOfProjection = {ShapeOfProjection}\n",
            f"PercentageDropout = {PercentageDropout}\n",
            f"Decoder = {Decoder}\n",
            f"ShapeOfDenseLayer = {ShapeOfDenseLayer}\n",
            f"Activation = {Activation}\n",
            f"Replicate = {Replicate}\n",
            
            f"NumberHeads = {NumberHeads}\n",
            "\n\n"]
            
            f.writelines(chrom_info)
        raise exc
        
    return model, vectorize_layer, vocab_list


#%% Main
RESULTS: str = "Article GA Results/Results 11/"
BEST_CHROM: str = "Best_chromosome.pkl"

with open(RESULTS + BEST_CHROM,'rb') as f:
    best_chromos = pickle.load(f)

chromosome = best_chromos[-1].copy()

decoded_chromosome = decode_chromosome(chromosome, verbose=True)

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

kf = StratifiedKFold(n_splits=NumberFolds, shuffle=True)
data_groups = kf.split(DATA, DATA['rating'])
for fn, (tr,te) in enumerate(data_groups):
    tf.keras.backend.clear_session() # Keras starts with a blank state at each iteration.
    
    if False: # For testing
        fn = 0; (tr,te) = next(data_groups)
    
    if SAMPLING_METHOD == 0: # Normal
    
        # Define datasets
        df_train = DATA.iloc[tr]
        df_train, df_val  = train_test_split(df_train, test_size=0.25,
                                            shuffle=True, stratify=df_train['rating'])
        df_test = DATA.iloc[te]
        # Extract features:
        XTrain = df_train['text']
        XValid = df_val['text']
        XTest  = df_test['text']
        
        # Extract labels:
        YTrain = df_train['rating']
        YValid = df_val['rating']
        YTest  = df_test['rating']
        
        # Apply cost-sensitive learning
        cw = create_classWeight(YTrain)
        
        # Apply One-Hot encoding
        encodeYTrain = np_utils.to_categorical(YTrain, NUMBER_CLASSES)
        encodeYValid = np_utils.to_categorical(YValid, NUMBER_CLASSES)
        
        if VerboseData:
            print("Full dataset:", len(DATA), '| 100%')
            trainSize = len(df_train)
            validSize = len(df_val)
            testSize  = len(df_test)
            
            datasetSize = trainSize + validSize + testSize
            trainSize /= datasetSize
            validSize /= datasetSize
            testSize  /= datasetSize
            print('Data Partitions: '
                  f'TRAIN {trainSize*100:.2f} ',
                  f'| VALID {validSize*100:.2f} ',
                  f'| TEST {testSize*100:.2f} |',)
            
    elif SAMPLING_METHOD == 1: # Under and Over sampling
    
        df_train, df_val  = train_test_split(DATA.iloc[tr], test_size=0.25,
                                            shuffle=True, stratify=DATA.iloc[tr]['rating'])
        df_train = under_sample(df_train, full_balance=True)
        df_test = DATA.iloc[te]
        
        # # Define datasets
        # df_train = DATA.iloc[tr]
        # df_train, df_val  = train_test_split(df_train, test_size=0.25,
        #                                     shuffle=True, stratify=df_train['rating'])
        # df_test = DATA.iloc[te]
        
        # Extract features:
        XTrain = df_train['text']
        XValid = df_val['text']
        XTest  = df_test['text']
        
        # Extract labels:
        YTrain = df_train['rating']
        YValid = df_val['rating']
        YTest  = df_test['rating']
        
        # Apply cost-sensitive learning
        cw = create_classWeight(YTrain)
        
        # Apply One-Hot encoding
        encodeYTrain = np_utils.to_categorical(YTrain, NUMBER_CLASSES)
        encodeYValid = np_utils.to_categorical(YValid, NUMBER_CLASSES)
        
        if VerboseData:
            print("Full dataset:", len(DATA), '| 100%')
            trainSize = len(df_train)
            validSize = len(df_val)
            testSize  = len(df_test)
            
            datasetSize = trainSize + validSize + testSize
            trainSize /= datasetSize
            validSize /= datasetSize
            testSize  /= datasetSize
            print('Data Partitions: '
                  f'TRAIN {trainSize*100:.2f} ',
                  f'| VALID {validSize*100:.2f} ',
                  f'| TEST {testSize*100:.2f} |',)
    
    elif SAMPLING_METHOD == 2: # Only Undersampling
        pass
    
    model, vectorize_layer, vocab_list = chromossome_to_architecture(decoded_chromosome, XTrain)
    
    print("==================================",
          "Tokenization layer...", sep='\n')
    t = time()
    trainData = vectorize_layer(XTrain)
    validData = vectorize_layer(XValid)
    testData = vectorize_layer(XTest)
    print(f"(Elapsed time: {strftime('%M:%S', gmtime(time()-t))})",
          "==================================", sep='\n')
        
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                  metrics=['accuracy', 'AUC'])

    # Define Callbacks
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc",
                                          patience=PatienteceValue,
                                          min_delta=min_delta,
                                          verbose=1,
                                          mode='max',
                                          restore_best_weights=True)
    sch = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    # TRAINING ******************************************
    print("Training started...")
    t = time()
    history = \
        model.fit(trainData, encodeYTrain,
        batch_size=BATCH_SIZE, epochs=EpochsValue,
        validation_data=(validData, encodeYValid),
        verbose=VerboseTrain, class_weight=(cw if SAMPLING_METHOD==0 else None), 
        callbacks=[es, sch])
    print("Training finished...")
    foldTime[fn] = time()-t
    bestEpoch[fn] = es.best_epoch+1
    
    # Plot training:
    historyACC.append((history.history['accuracy'],history.history['val_accuracy']))
    historyAUC.append((history.history['auc'],history.history['val_auc']))
    historyLOSS.append((history.history['loss'],history.history['val_loss']))
    
    show_train_history(historyACC[-1],'acc', bestEpoch[fn], '')
    show_train_history(historyAUC[-1],'auc', bestEpoch[fn], '')
    show_train_history(historyLOSS[-1],'loss', bestEpoch[fn], '')
    # *******************************************************

    # Test model
    testForecast = model.predict(testData)
    testYhat = np.argmax(testForecast, axis=-1)
    YTest = YTest.to_numpy()
    
    cmTest = ConfusionMatrix(YTest, testYhat, is_imbalanced=True)
    # classTest = classification_report(YTest, testYhat) #!!!
    ovr_AUC = metrics.roc_auc_score(YTest, testForecast, average='macro', multi_class='ovr')
    # class_AUC = metrics.roc_auc_score(YTest, testForecast, average=None, multi_class='ovr')
    # class_AUC = np.array([0., 0., 0.])
    foldAUC[fn] = ovr_AUC
    foldACC[fn] = cmTest.Overall_ACC
    foldSpe[fn] = cmTest.TNR_Macro
    foldSen[fn] = cmTest.TPR_Macro
    foldF1[fn] = cmTest.F1_Macro
    foldCM[fn] = cmTest.to_array(normalized=True)

    # Store metrics (assuming python dictionary is ordered)
    LabelsOverall = [k for k,v in cmTest.overall_stat.items() if isinstance(v,(float,int)) ]
    LabelsClass   = [k for k,v in cmTest.class_stat.items() if isinstance(v[0],(float,int))]
    
    foldOverall[fn] = selectMetrics(cmTest.overall_stat,LabelsOverall) +[ovr_AUC]
    foldClass[fn]   = np.asarray([list(d.values()) for d in selectMetrics(cmTest.class_stat,LabelsClass)])
    # AUC_id = 36
    # foldClass[fn, AUC_id] = class_AUC

#%% Save

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
model_name = "BC_up_to_Gen11_5fold" 
print("Model name to save:", model_name)

saveModels.save([model], model_name, '', path=MetricsPath,
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

with open(MetricsPath + f'/Model_{model_name}/Training/TrainingCurves', 'wb') as f:
    # Saves training curves as dict
    pickle.dump({'ACC':historyACC,
                 'AUC':historyAUC,
                 'LOSS':historyLOSS}, f)
    
#%% Notify me
from botNotify import TelegramBot

try:
    bot = TelegramBot()
    bot.telegram_bot_sendtext("Evaluation of model finished! ",
                              modelName=model_name.replace('_','-'),
                              Time=strftime('%H:%M:%S', gmtime(np.sum(foldTime))),
                              AUC=f"{np.mean(foldAUC)*100:.2f} %",
                              F1=f"{np.mean(foldF1)*100:.2f} %",
                              ACC=f"{np.mean(foldACC)*100:.2f} %",
                              Spe=f"{np.mean(foldSpe)*100:.2f} %",
                              Sen=f"{np.mean(foldSen)*100:.2f} %")
except:
    print(f"Finished model {model_name.replace('_','-')} but couldn't notify due to failed connection!")










