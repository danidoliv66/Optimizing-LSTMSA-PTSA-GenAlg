# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:48:47 2024

@author: Daniel Parada
"""
#%% Parameters
# *****************************************************************************
dataPath: str    = './Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv'
Vocab_Size: int  = 11560   # 11560 is value used in Thesis
max_seq_len: int = 100
vector_size: int = 512
BATCH_SIZE: int  = 128
min_delta: float = 0.0005 
patience: int    = 37
NN_epochs: int   = 150
verbose: int     = 1
# *****************************************************************************
#%% Control variables
# ***************************************************************************** 
verbose: bool = True
NumberFolds: int = 5
NumberOverall: int = 53 # Pycm documentation (only int and float)
NumberClass: int   = 55 # Pycm documentation (only int and float)

save: bool = False
# model_name: str = "sALSTM_Baseline_FC_v0" # self-Attention LSTM (Baseline) + Full Cleaning + Blankspace
model_name: str = "sALSTM_Baseline_FC_v1" # self-Attention LSTM (Baseline) + Full Cleaning + Wordpiece # (PROPER BASELINE)
# model_name: str = "sALSTM_Baseline_FC_v1.1" # self-Attention LSTM (Baseline) + Full Cleaning + Wordpiece + fixed posEmbedding # (NO)

savePath: str = './TrainedModels&Metrics'
#%% Load libraries
# *****************************************************************************
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
from time import time, strftime, gmtime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide INFO messages
# Keras libraries:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding
from keras_nlp.tokenizers import WordPieceTokenizer
from tensorflow.keras.layers import Bidirectional as Bi

# Local libraries:
from Models import saveModels
from Tokenizer.VocabBuilder import Word2VecVocabulary, WordPieceVocabulary
from tools import merge_pos_neg_neutral
from Cleaning.textPreprocessing import clean, clean_tokens, FC
from Models.getMetrics import show_train_history, selectMetrics
from main.selfAttention import (TokenAndPositionEmbedding,
                                build_pos_matrix,
                                MHAttention)
#%% Functions
def json_to_tokens():
    """parse json file to list of list (not padded)"""
    with open("Corpus/tokenized_data.json", 'r', encoding='utf-8') as f:
        tokenized_data = json.load(f)
        
    review_ids = list(map(int,tokenized_data.keys()))
    tokens = list(tokenized_data.values())
        
    return review_ids, tokens

def get_reviews(texts, review_ids, ratings) -> list:
    review_ids = pd.Series(review_ids) 
    mask = pd.Series(review_ids).isin(ratings['review_id'])
    idxs = review_ids[mask].index
    
    text_set = []
    for i in idxs:
        text_set.append(texts[i])
    return text_set

def get_ratings(ratings) -> list:
    return ratings['rating']

def create_df(review_ids, texts, ratings):
    # print(len(review_ids), len(texts), len(ratings),)
    # for i in range(len(review_ids)):
    #     if review_ids[i] != ratings.review_id.iloc[i]:
    #         print("ID do not match!")
    #         break
    # print("review_ids matches in both ´texts´ and ´ratings´")
    df = pd.concat([ratings, texts], 
                    axis=1, ignore_index=True).rename({0:'review_id',1:'rating',2:'text'},axis=1)
    return df

def create_classWeight(tr, n_classes=3):
    n_samples = len(tr)
    wj = {}
    for j in range(n_classes):
        wj[j] = n_samples/(n_classes*len(tr[tr==j]))
    print("class weights = ",wj)
    return wj

def schedule(epoch, lr):
    if epoch < 3: return lr # warm-up
    if lr < 1e-5: return lr # limiter
    else: return lr * tf.math.exp(-0.1) # lr decay
    
# *****************************************************************************
#%% Load data
# *****************************************************************************
review_ids, texts = json_to_tokens()
texts = pd.Series(list(map(lambda x: clean(' '.join(x), *FC), texts)))
review_ids = pd.Series(review_ids)
ratings = merge_pos_neg_neutral(pd.read_csv(dataPath, usecols=['review_id','rating']))

df = create_df(review_ids, texts, ratings)
# *****************************************************************************
#%% Cross validation:
# *****************************************************************************
kf = StratifiedKFold(n_splits=NumberFolds, shuffle=True)
data_groups = kf.split(ratings, ratings['rating'])
    
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
for fn, (tr,te) in enumerate(data_groups):
    if False: # For testing
        fn = 0; (tr,te) = next(data_groups)
        
    print("fold:", fn)
    # Create data partitions
    df_train = df.iloc[tr]
    df_train, df_val = train_test_split(df_train, test_size=0.25, # 0.125,
                                        shuffle=True, stratify=df_train['rating'])
    df_test   = df.iloc[te]
    # del tr, te
        
    trainSize = len(df_train)
    validSize = len(df_val)
    testSize  = len(df_test)
    log[fn] += f"Data size used: TRAIN {trainSize}, TEST {testSize}, VALIDATION {validSize}\n"
    if verbose:
        datasetSize = trainSize + validSize + testSize
        trainSize /= datasetSize
        validSize /= datasetSize
        testSize  /= datasetSize
        print('Data Partitions: '
              f'TRAIN {trainSize*100:.2f} ',
              f'| VALID {validSize*100:.2f} ',
              f'| TEST {testSize*100:.2f} |',) 
    
    # Get reviews:
    # train_reviews = get_reviews(texts, review_ids, df_train)
    # valid_reviews = get_reviews(texts, review_ids, df_val)
    # test_reviews = get_reviews(texts, review_ids, df_test)
    train_reviews = df_train['text']
    valid_reviews = df_val['text']
    test_reviews  = df_test['text']
    
    if verbose:
        trainSize = len(train_reviews) / datasetSize
        validSize = len(valid_reviews) / datasetSize
        testSize  = len(test_reviews) / datasetSize
        print('Data Partitions: '
              f'TRAIN {trainSize*100:.2f} ',
              f'| VALID {validSize*100:.2f} ',
              f'| TEST {testSize*100:.2f} |',)  
    
    # Get ratings:
    # train_ratings = get_ratings(df_train)
    # valid_ratings = get_ratings(df_val)
    # test_ratings = get_ratings(df_test)
    train_ratings = df_train['rating']
    valid_ratings = df_val['rating']
    test_ratings = df_test['rating']
    
    if verbose:
        trainSize = len(train_ratings) / datasetSize
        validSize = len(valid_ratings) / datasetSize
        testSize  = len(test_ratings) / datasetSize
        print('Data Partitions: '
              f'TRAIN {trainSize*100:.2f} ',
              f'| VALID {validSize*100:.2f} ',
              f'| TEST {testSize*100:.2f} |',) 
        
    # Set CSL using class_weight parameter:
    cw = create_classWeight(train_ratings)
    # One-hot encoding on labels:
    train_ratings = tf.keras.utils.to_categorical(train_ratings)
    valid_ratings = tf.keras.utils.to_categorical(valid_ratings)
    test_ratings  = tf.keras.utils.to_categorical(test_ratings)
    
    # Train Embedding Matrix using Word2Vec
    # *************************************************************************
    if 'v0' in model_name:
        wv, vocab_list, embedding_matrix = \
        Word2VecVocabulary(corpus=train_reviews,
                           vector_size=vector_size,
                           maxlen=max_seq_len,
                           vocab_size=Vocab_Size)
        
        vocab_size = len(vocab_list)+2
        print("Vocabulary size:", vocab_size)
        vectorize_layer = layers.TextVectorization(max_tokens=vocab_size,standardize=None,
                                                    output_mode='int',
                                                    output_sequence_length=max_seq_len,
                                                    vocabulary=vocab_list)
        #Fullfill vocabulary:
        for token in ['[UNK]','[PAD]']: # include new tokens #!!!
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,vector_size))
        UNKvector = np.zeros((1,vector_size))
        embedding_matrix = np.concatenate([PADvector, UNKvector, embedding_matrix])
        
        vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                             sequence_length=max_seq_len)
    elif 'v1' in model_name:
        wv, vocab_list, embedding_matrix = \
        WordPieceVocabulary(train_reviews, 
                            vector_size=vector_size, 
                            maxlen=max_seq_len, 
                            vocab_size=Vocab_Size)
        vocab_size = len(vocab_list)+2
        
        #Fullfill vocabulary:
        for token in ['[UNK]','[PAD]']: # include new tokens #!!!
            vocab_list.insert(0,token)
        # Add Special tokens to embedding matrix:
        PADvector = np.zeros((1,vector_size))
        UNKvector = np.zeros((1,vector_size))
        embedding_matrix = np.concatenate([PADvector, UNKvector, embedding_matrix])
        
        vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                             sequence_length=max_seq_len)
    # *************************************************************************
    # Tokenization
    # *************************************************************************
    print("Tokenization layer...")
    t = time()
    trainData = vectorize_layer(train_reviews)
    validData = vectorize_layer(valid_reviews)
    testData = vectorize_layer(test_reviews)
    print(f"(Elapsed time: {strftime('%M:%S', gmtime(time()-t))})")
    # *************************************************************************
    # Define model architecture #!!!
    # *************************************************************************
    n_timesteps = max_seq_len
    n_features = vector_size
    model_dim = vector_size
    dim_proj = 64
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    loss_func = loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    metrics = ['accuracy', 'AUC']
    
    # Define layers: (Based on Xia)
    position_matrix = build_pos_matrix(n_timesteps, vector_size)
    
    if 'v1.1' in model_name:
        Embed_layer = Embedding(vocab_size,vector_size,
                                input_length=n_timesteps,
                                weights=[embedding_matrix],
                                mask_zero=True,
                                name='TokenEmbedding')
    else:
        Embed_layer = \
        TokenAndPositionEmbedding(vocab_size, vector_size, maxlen=n_timesteps,
                                  weights_tokens=embedding_matrix,
                                  weights_position=position_matrix,
                                  mask_zero=True, name='TokenEmbedding')
    
    firstRecurssiveEncoder = Bi(layers.LSTM(model_dim//2 ,return_sequences=True,
                                            dropout=0.1, kernel_regularizer=regularizers.l2(0.0001)))
    secondRecurssiveEncoder = Bi(layers.LSTM(model_dim//2 ,return_sequences=True,
                                             dropout=0.1, kernel_regularizer=regularizers.l2(0.0001)))
    
    attentionLayer = MHAttention(heads=model_dim//dim_proj, 
                                 dim_K=dim_proj, dim_V=dim_proj, 
                                 d_model=model_dim, activation='tanh', name='Attention')

    recurrentDecoder = layers.LSTM(model_dim, return_sequences=False, 
                                   name='RecurrentDecoder')
    FFNN_layer = layers.Dense(model_dim//2, activation=None)
    FFNN_activation = layers.Activation('relu')
    Classifier = layers.Dense(3, activation=None)
    Output_layer = layers.Activation('softmax')
    
    # Build the Neural Network:
    model_inputs = layers.Input(shape=(n_timesteps), name='model_inputs')
    mask_input = Embed_layer.compute_mask(model_inputs)
    embed_out = Embed_layer(model_inputs)
    embed_out = layers.Dropout(0.1)(embed_out)
    
    enc_out1 = firstRecurssiveEncoder(embed_out)
    encoder_out = secondRecurssiveEncoder(enc_out1)
    
    if 'v1.1' in model_name:
        encoder_out = encoder_out + position_matrix
    
    attention_out = attentionLayer(encoder_out,encoder_out,
                                   attention_mask=mask_input)
    decoder_out = recurrentDecoder(attention_out)
    
    ffnn_out = FFNN_layer(decoder_out)
    ffnn_out = FFNN_activation(ffnn_out)
    ffnn_out = layers.Dropout(0.1)(ffnn_out)
    classifier_out = Classifier(ffnn_out)
    model_outputs = Output_layer(classifier_out)
    
    model = Model(model_inputs, model_outputs, name=model_name)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=metrics)
    # *************************************************************************
    # Define Callbacks:
    # *************************************************************************
    es = keras.callbacks.EarlyStopping(monitor="val_auc", patience=patience,
                                       min_delta=min_delta, verbose=1, mode='max',
                                       restore_best_weights=True)
    sch = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    callbacks = [es, sch]
    # *************************************************************************
    # Train the model:
    # *************************************************************************
    print("Training started...")
    t = time()
    history = model.fit(trainData,train_ratings,
                        validation_data=(validData,valid_ratings),
                        epochs=NN_epochs, batch_size=BATCH_SIZE,
                        callbacks=callbacks, class_weight=cw,
                        verbose=verbose)
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
    
    # Evaluate current fold
    # *************************************************************************
    print("\nTesting model...")
    test_real_labels = np.argmax(test_ratings, axis=-1)
    
    test_forecasts = model.predict(testData)
    # test_forecasts = model.predict([testData,testMask])
    test_y_pred = np.argmax(test_forecasts, axis=-1)
    
    classTest = classification_report(test_real_labels, test_y_pred)
    cmTest = ConfusionMatrix(test_real_labels, test_y_pred,is_imbalanced=True)
    ovr_AUC = roc_auc_score(test_ratings, test_forecasts, average='macro',multi_class='ovr')
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
    foldClass[fn,AUC_id] = roc_auc_score(test_ratings, test_forecasts, average=None, multi_class='ovr') # use sklearn AUC
    # *************************************************************************
    # Restore session before creating a new model
    tf.keras.backend.clear_session()
# *****************************************************************************
#%% Print Log
full_log = []
for f in range(NumberFolds):
    print(f"Log of fold #{f+1}:",log[f],sep='\n')
    full_log.extend([60*'=' + '\n',log[f]])
full_log.append(60*'=' + '\n')
# *****************************************************************************
#%% Save model
# *****************************************************************************
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
    
    with open(f'{savePath}/Model_{model_name}/trainLog.txt','w') as log_file:
        log_file.writelines(full_log)
# *****************************************************************************
