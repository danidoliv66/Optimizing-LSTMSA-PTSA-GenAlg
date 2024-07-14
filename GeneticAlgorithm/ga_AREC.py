# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:41:05 2023
@author: danie

Files description:
    "Population0.txt" -> store full population of generation 0
    "Score0.txt" -> store full population scores of generation 0
    "PopulationParentsBegin.pkl" -> store full population of each generation, sequentially
    "ScoreParentsBegin.pkl" -> store full population scores of each generation, sequentially
    
    NOTE: (This is not true)
        Content of Population0.txt MUST match with first generation of PopulationParentsBegin
        Content of Score0.txt MUST match with first generation of ScoreParentsBegin
    NOTE:
        PopulationParentsBegin is written at the end of each fitness score, and saves the 
        initial population of that fitness function
        
    "Population{gen}End.txt -> store full population of gen
    "Score{gen}End.txt" -> store full populationscore of gen
     This population consists of parents (Elitism) and children, and are sorted beggining with best
     
    "Best_score.pkl" -> store best population (first of list) of each generation
    "Best_chromosome.pkl" -> store best population score (first of list) of each generation
    
    NOTE:
        Best_chromosome.pkl is continuously overwritten by global variable best_chromosome
        Best_score.pkl is continuously overwritten by global variable best_score
    
"""
#%% Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide INFO messages

import gc
import json 
import pickle
import random
import itertools
from pathlib import Path
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
from scipy.spatial import distance

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
from Models.getMetrics import show_train_history
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

def _exception_msg(msg: str):
    print(30*'*', 
          msg, '', 
          30*'*', sep='\n')

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
    
# Class to clear memory during Training
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

#%% Load data
# *****************************************************************************
dataPath: str    = './Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv'
review_ids, texts = json_to_tokens()
texts = pd.Series(list(map(lambda x: clean(' '.join(x), *FC), texts)))
review_ids = pd.Series(review_ids)
ratings = merge_pos_neg_neutral(pd.read_csv(dataPath, usecols=['review_id','rating']))

DATA: pd.DataFrame = create_df(review_ids, texts, ratings)
# *****************************************************************************
#%% Control variables
NumberFolds:     int = 2
PatienteceValue: int = 37
EpochsValue:     int = 150
min_delta:     float = 0.0005
# Verbose
VerboseData:   int = 1
VerboseTrain:  int = 1
VerboseChrom: bool = True
VerboseGen:   bool = True

fromCheckpoint: bool = True
MetricsPath:     str = "./GeneticAlgorithm/Results/"

#%% Constants
NUMBER_CLASSES:   int   = 3      # number of classes to consider for classification
NUMBER_TIMESTEPS: int   = 100    # number of tokens to consider for model input
BATCH_SIZE:       int   = 128    # Batch size on GPU
VOCAB_SIZE:       int   = 11_560 # vocabulary size of model input
LEARNING_RATE:    float = 0.0001  # learning rate of RMSProp optimizer

#%% Functions definition   #!!!
def initilization_of_population(size: int, n_bits: int) -> np.array:
    """
    Method for initializing the population
    Creates a set of randomly initialized chromossomes
    """
    population = []
    
    # Include Baseline Chromossome: ("sALSTM_Baseline_FC_v1")
    # Already verified: Resultant model is exactly the same as Baseline!
    population.append(np.array([1.,       # Bidirectional LSTM
                                0.,1.,    # 2 stacked Bi-LSTM layers
                                1.,0.,    # Attention after LSTMs
                                0.,1.,    # 512 Model dimension
                                0.,1.,    # 64 inner projection
                                0.,0.,1., # 10% Dropout
                                1.,0.,    # LSTM Decoder
                                0.,1.,    # 512//2 Classifier dimension
                                0.,0.,    # ReLU activation function
                                0.,0.,    # Do not replicate block
                                1.,       # WordPiece tokenizer
                               ])
                      )
    
    for i in range(size-1):
        chromosome = np.ones(n_bits)
        chromosome[:(n_bits//2)] = 0
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return np.array(population)

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
        
    return model, vectorize_layer


def fitness_score(population: list, gen: int) -> (np.array, np.array):
    """
    Method for defining the fitness function to optimize
    @param population: list of chromossomes
    
    This function is called for the first time directly to set the first generation
    characteristics. Then it's only called inside generations() functions

    This function compiles each model dictated by the cromossomes and returns (?)
    
    """
    #*****************************************************
    if False: # For Testing
        population = initilization_of_population(2, 21)
        gen = fn = id_pop = 0
        chromosome = population[0]
        print(chromosome)
    #*****************************************************
        
    print(f"Fitness score of Generation {gen}\n")
    scores = [] # Stores the meanAUC after each chromosome
    for id_pop, chromosome in enumerate(population):

        print(f"\n[Gen{gen}|Chrom{id_pop+1}]:","===========================================", sep='\n')
        decoded_chromosome = decode_chromosome(chromosome, verbose=VerboseChrom)
        print("===========================================\n")
        
        # Variables to store metrics
        AccAtEnd = np.zeros(NumberFolds)
        AUCAtEnd = np.zeros(NumberFolds)
        SenAtEnd = np.zeros(NumberFolds)
        SpeAtEnd = np.zeros(NumberFolds)
        F1sAtEnd = np.zeros(NumberFolds)
            
        kf = StratifiedKFold(n_splits=NumberFolds, shuffle=True)
        for fn,(tr,te) in enumerate(kf.split(DATA, DATA['rating'])):
            
            #***************************************************
            if False: # For testing
                (tr,te) = next(kf.split(DATA, DATA['rating']))
            #***************************************************
                
            gc.collect()
            tf.keras.backend.clear_session() # Keras starts with a blank state at each iteration.
            if VerboseGen: print(f"\n[Gen{gen}|Chrom{id_pop+1}] Cross Validation, fold {fn}\n")
            # Define datasets
            df_train = DATA.iloc[tr]
            df_test, df_val  = train_test_split(DATA.iloc[te],test_size=0.40,
                                                shuffle=True,stratify=DATA.iloc[te]['rating'])
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
            
            model, vectorize_layer = chromossome_to_architecture(decoded_chromosome, XTrain)

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
            # continue #!!!
            
            try: # TRAINING ******************************************
                
                print("Training started...")
                t = time()
                history = \
                model.fit(trainData, encodeYTrain,
                          batch_size=BATCH_SIZE, epochs=EpochsValue,
                          validation_data=(validData, encodeYValid),
                          verbose=VerboseTrain, class_weight=cw, 
                          callbacks=[es, sch, ClearMemory()])
                print("Training finished...")
                trainTime = time() - t
                
                # Plot training:
                show_train_history((history.history['auc'],history.history['val_auc']),
                                   'auc', es.best_epoch+1, 'model')
                show_train_history((history.history['loss'],history.history['val_loss']),
                                   'loss', es.best_epoch+1, 'model')
            # *******************************************************
            except tf.errors.ResourceExhaustedError as e:
                with open(MetricsPath + 'logErrors.txt','a') as f:
                    header = "[log at {}]: Error training the model: {}\n". \
                        format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), e)
                    to_write = \
                    [header,
                     'Chromosome: ' + ''.join([str(int(c)) for c in chromosome]) + '\n',
                     'Fold: ' + str(fn) + '\n',
                     '\n\n']
                    f.writelines(to_write)
                # Set all metrics to zero
                AccAtEnd = np.zeros(NumberFolds)
                AUCAtEnd = np.zeros(NumberFolds)
                SenAtEnd = np.zeros(NumberFolds)
                SpeAtEnd = np.zeros(NumberFolds)
                F1sAtEnd = np.zeros(NumberFolds)
                break # breaks cross-validation loop
                
            except KeyboardInterrupt:
                with open(MetricsPath + 'logErrors.txt','a') as f:
                    header = "[log at {}]: Training stopped by user\n". \
                        format(strftime('%H:%M:%S, %d/%b/%Y', localtime()))
                    to_write = [header]
                    f.writelines(to_write)
                # Set all metrics to zero
                AccAtEnd = np.zeros(NumberFolds)
                AUCAtEnd = np.zeros(NumberFolds)
                SenAtEnd = np.zeros(NumberFolds)
                SpeAtEnd = np.zeros(NumberFolds)
                F1sAtEnd = np.zeros(NumberFolds)
                raise KeyboardInterrupt()
            
            # Test model
            testForecast = model.predict(testData)
            testYhat = np.argmax(testForecast, axis=-1)
            YTest = YTest.to_numpy()

            cmTest = ConfusionMatrix(YTest, testYhat,is_imbalanced=True)
            AccAtEnd[fn] = cmTest.Overall_ACC
            SenAtEnd[fn] = cmTest.TPR_Macro
            SpeAtEnd[fn] = cmTest.TNR_Macro
            AUCAtEnd[fn] = metrics.roc_auc_score(YTest, testForecast, average='macro',multi_class='ovr')
            F1sAtEnd[fn] = cmTest.F1_Macro
            
            if VerboseGen: 
                table_row = "{:>10} {:>10} {:>10} {:>10} {:>10}"
                f = lambda x: round(x,2) if isinstance(x,(float, int)) else str(x)
                support = sum(list(cmTest.P.values()))
                classTest = [
                    table_row.format('', 'precision','recall','f1-score','support'),
                    '',
                    table_row.format('0',f(cmTest.PPV[0]),f(cmTest.TPR[0]),f(cmTest.F1[0]),f(cmTest.P[0])),
                    table_row.format('1',f(cmTest.PPV[1]),f(cmTest.TPR[1]),f(cmTest.F1[1]),f(cmTest.P[1])),
                    table_row.format('2',f(cmTest.PPV[2]),f(cmTest.TPR[2]),f(cmTest.F1[2]),f(cmTest.P[2])),
                    '',
                    table_row.format('accuracy','','',f(cmTest.Overall_ACC),support),
                    table_row.format('macro avg',f(cmTest.PPV_Macro),f(cmTest.TPR_Macro),f(cmTest.ACC_Macro),support),
                    ]
                print("Classification report TEST",*classTest,sep='\n')   
                print(f"\nTraining time: {strftime('%H:%M:%S', gmtime(trainTime))}\n")
                
            with open(MetricsPath + "TimePerPop.txt", 'a') as f:
                to_write = [f"\n[Gen{gen}|Chrom{id_pop+1}] fold{fn} --> {strftime('%H:%M:%S', gmtime(trainTime))}\n"]
                f.writelines(to_write)
            
            del model, vectorize_layer
#########################################################################################################################################
        # Calculate metrics combining K folds
        AvgACC = np.mean(AccAtEnd) * 100
        AvgSen = np.mean(SenAtEnd) * 100
        AvgSpe = np.mean(SpeAtEnd) * 100
        AvgAUC = np.mean(AUCAtEnd) * 100
        AvgF1s = np.mean(F1sAtEnd) * 100
        
        StdACC = np.std(AccAtEnd) * 100
        StdSen = np.std(SenAtEnd) * 100
        StdSpe = np.std(SpeAtEnd) * 100
        StdAUC = np.std(AUCAtEnd) * 100
        StdF1s = np.std(F1sAtEnd) * 100
            
        if VerboseGen:
            print("\n===================================",
                  f"== Gen{gen:0>2} | Chrom{id_pop+1:0>2} | Fold{fn} ========",
                  'Final results:', sep='\n')
            print(f'-> Accuracy: Avg({AvgACC:.2f}%), Std({StdACC:.2f}%)')
            print(f'-> Sensitivity: Avg({AvgSen:.2f}%), Std({StdSen:.2f}%)')
            print(f'-> Specificity: Avg({AvgSpe:.2f}%), Std({StdSpe:.2f}%)')
            print(f'-> AUC: Avg({AvgAUC:.2f}%), Std({StdAUC:.2f}%)')
            print(f'-> F1-score: Avg({AvgF1s:.2f}%), Std({StdF1s:.2f}%)')
            print("===================================\n")
            
        if id_pop == 0:
            header = f"[New generation: {gen}]\nPopulation {id_pop+1:0>2}: "
        else:
            header = f"Population {id_pop+1:0>2}: "
            
        with open(MetricsPath + "ACC.txt", 'a') as f:
            f.write(header)
            f.write(f"{AvgACC:.3f} ± {StdACC:.4f}\n")
        with open(MetricsPath + "Sen.txt", 'a') as f:
            f.write(header)
            f.write(f"{AvgSen:.3f} ± {StdSen:.4f}\n")
        with open(MetricsPath + "Spe.txt", 'a') as f:
            f.write(header)
            f.write(f"{AvgSpe:.3f} ± {StdSpe:.4f}\n")
        with open(MetricsPath + "AUC.txt", 'a') as f:
            f.write(header)
            f.write(f"{AvgAUC:.3f} ± {StdAUC:.4f}\n")
        with open(MetricsPath + "F1.txt", 'a') as f:
            f.write(header)
            f.write(f"{AvgF1s:.3f} ± {StdF1s:.4f}\n")

        scores.append(AvgAUC) # here I decide what metric I want to consider
    
    # Convert into arrays
    scores = np.array(scores)
    # scores = np.random.normal(0.90,0.04,size=(len(population))) #!!!
    population = np.array(population)

    # sort and reverse order of based on `scores`
    inds = np.flip(np.argsort(scores)) # first is best, last is worst

    scoresGenTemp = np.zeros((NumbParents))
    populationGenTemp = np.zeros((NumbParents, NumbBitsDecimal))
    for i,g in enumerate(inds):
        scoresGenTemp[i] = scores[g]
        populationGenTemp[i, :] = population[g, :]

    if VerboseGen: print(f"Finished Fitness score for Generation {gen}")
    return scoresGenTemp, populationGenTemp

def selection(popParents: np.array, popChild: np.array, 
              scoresParents: np.array, scoresChild: np.array,
              ElitsNumber: int):
    if VerboseGen: print("Selecting next population...")
    NumbParents, NumbBitsDecimal = popParents.shape
    # Group Parents & Children (excluding Elitism)
    population_Final = np.zeros(((NumbParents * 2) - ElitsNumber, NumbBitsDecimal))
    for LinesAppend in range(0, (NumbParents * 2) - ElitsNumber, 1):
        if LinesAppend < NumbParents - ElitsNumber:
            population_Final[LinesAppend, :] = popParents[LinesAppend + ElitsNumber, :].copy()
        else:
            population_Final[LinesAppend, :] = popChild[LinesAppend - NumbParents + ElitsNumber, :].copy()
    scoresGenFinal = np.append(scoresParents[ElitsNumber:], scoresChild, axis=0)
    
    # Sort Parents & Children (best ones are first)
    inds = np.flip(np.argsort(scoresGenFinal))
    scoresGenTemp = np.zeros(((NumbParents * 2) - ElitsNumber))
    populationGenTemp = np.zeros(((NumbParents * 2) - ElitsNumber, NumbBitsDecimal))
    for sortingLines in range(0, len(inds), 1):
        scoresGenTemp[sortingLines] = scoresGenFinal[inds[sortingLines]].copy()
        populationGenTemp[sortingLines, :] = population_Final[inds[sortingLines], :].copy()

    scoresGenUpdate = np.zeros((NumbParents))
    populationGenUpdate = np.zeros((NumbParents, NumbBitsDecimal))
    # Select elitism
    for k in range(0, ElitsNumber):
        scoresGenUpdate[k] = scoresParents[k].copy()
        populationGenUpdate[k, :] = popParents[k, :].copy()
    # Select rest of population, starting from best
    scoresGenUpdate[ElitsNumber:NumbParents] = scoresGenTemp[0:NumbParents - ElitsNumber].copy()
    populationGenUpdate[ElitsNumber:NumbParents, :] = populationGenTemp[0:NumbParents - ElitsNumber, :].copy()
    
    # Sort Selected population (best ones are first)
    inds = np.flip(np.argsort(scoresGenUpdate))
    scoresGenUpdateSorted = np.zeros((NumbParents))
    populationGenUpdateSorted = np.zeros((NumbParents, NumbBitsDecimal))
    for sortingLines in range(0, len(inds), 1):
        scoresGenUpdateSorted[sortingLines] = scoresGenUpdate[inds[sortingLines]].copy()
        populationGenUpdateSorted[sortingLines, :] = populationGenUpdate[inds[sortingLines], :].copy()
    
    return scoresGenUpdateSorted, populationGenUpdateSorted
    
def crossover(pop_after_sel: np.array, cross_Rate: float) -> np.array:
    """
    Method for creating the crossover
    This function is called inside generations() function
    """
    if VerboseGen: print("Applying crossing over on chromosomes...")
    NumbBitsDecimal = pop_after_sel.shape[-1]
    population_nextgen_crossover = []
    for _ in range(len(pop_after_sel)): # Run over all chromosomes (whole population)
        if random.random() <= cross_Rate:
            while True:
                # Select 2 random chromosomes from population (must be different!)
                numbers1 = random.randrange(0, len(pop_after_sel))
                numbers2 = random.randrange(0, len(pop_after_sel))
                parent1 = min(numbers1, numbers2)
                numbers1 = random.randrange(0, len(pop_after_sel))
                numbers2 = random.randrange(0, len(pop_after_sel))
                parent2 = min(numbers1, numbers2)
                if parent1 != parent2:
                    break
            # Select random points in chromosome for crossing over
            CrossoverPoints1 = random.randrange(0, NumbBitsDecimal)
            CrossoverPoints2 = random.randrange(0, NumbBitsDecimal)
            CrossoverPoints = np.sort([CrossoverPoints1, CrossoverPoints2+1])
            child = pop_after_sel[parent1].copy()
            child[CrossoverPoints[0]:CrossoverPoints[1]] = pop_after_sel[parent2,CrossoverPoints[0]:CrossoverPoints[1]]
            
        else: # Don't apply crossing over
        
            # Select random chromosome from population
            numbers1 = random.randrange(0, len(pop_after_sel))
            numbers2 = random.randrange(0, len(pop_after_sel))
            parent1 = min(numbers1, numbers2)
            child = pop_after_sel[parent1].copy()
            
        population_nextgen_crossover.append(child) 
    return np.array(population_nextgen_crossover) # have same length as input population


def mutation(pop_after_cross: np.array, MutationRate: float, gen: int) -> np.array:
    """
    Method for making mutation on the population
    This function is called inside generations() function
    """
    if VerboseGen: print("Applying mutations on chromosomes...")
    mutation_rate = MutationRate - MutationRate * int(gen / 5) * 0.3
    if mutation_rate < 0.01:
        mutation_rate = 0.01

    population_nextgen_mutation = []
    for chromosome in pop_after_cross: # iterates all chromosomes
        mutant = np.zeros(len(chromosome))
        for i in range(len(chromosome)):
            if random.random() <= mutation_rate:
                mutant[i] = 1 - chromosome[i]  # inverts the locus value
            else: # Don't apply mutation
                mutant[i] = chromosome[i]
                
        population_nextgen_mutation.append(mutant)
    return np.array(population_nextgen_mutation) # have same length as input population


def generations(NumbParents: int, 
                mutation_rate: float, crossover_rate: float, 
                gen: int, population_nextgen: np.array, scoresGen: np.array, 
                ElitsNumber: int,
                best_score: list, best_chromo: list) -> (np.array, np.array):
    """
    Method for creating the next generation
    """
    print(f"\n\nNew generation: {gen}\n\n")
    
    popParents = population_nextgen.copy()
    scoresParents = scoresGen.copy()
    
    # Save Initial population of current generation (used to keep track of current generation)
    with open(MetricsPath + f"Population{gen:0>2}Begin.txt", 'w') as f:
        f.write(str(popParents))
    # Save Initial score of current generation
    with open(MetricsPath + f"Score{gen:0>2}Begin.txt", 'w') as f:
        f.write('['+'\n '.join(scoresParents.astype('str'))+']')
    
    # Crossing over:
    pop_after_cross = crossover(popParents, crossover_rate)                    # CROSSOVER
    # Mutations:
    pop_after_mut = mutation(pop_after_cross, mutation_rate, gen)              # MUTATION
    
    # Fit population:
        
    # Save Initial population of current generation (for checkpoint restart)
    with open(MetricsPath + "PopulationParentsToFit.pkl", 'ab') as f:
        pickle.dump(pop_after_mut, f)
    
    scoresChild, popChild = fitness_score(pop_after_mut, gen)                  # FITNESS SCORE
    # Select population for next generation:
    scoresNextGen, populationNextGen = selection(popParents, popChild,         # SELECTION
                                                 scoresParents, scoresChild, 
                                                 ElitsNumber)
    
    
    # Save Initial score of current generation
    with open(MetricsPath + "ScoreParentsBegin.pkl", 'ab') as f:
        pickle.dump(scoresParents, f)
        
    # Save Final population of current generation (used to keep track of current generation)
    with open(MetricsPath + f"Population{gen:0>2}End.txt", 'w') as f:
        f.write(str(populationNextGen))
    # Save Final score of current generation
    with open(MetricsPath + f"Score{gen:0>2}End.txt", 'w') as f:
        f.write('['+'\n '.join(scoresNextGen.astype('str'))+']')

    best_score.append(scoresNextGen[0])
    best_chromo.append(populationNextGen[0])
    
    # Save all best chromosomes untill now
    with open(MetricsPath + "Best_score.pkl", 'wb') as f:
        pickle.dump(best_score, f)
    with open(MetricsPath + "Best_chromosome.pkl", 'wb') as f:
        pickle.dump(best_chromo, f)

    return scoresNextGen, populationNextGen, best_score, best_chromo

def compute_diversity(best_chromo=None, best_score=None, secondsGen=None, verbose=False):
    """
    This function computes diversity of each generation. Each generation is
    composed by:
        A population of NumbParents chromosomes,
        A score for each chromosome (not used here),
        The best chromosome of its population (paired with its best score),
        Processing time
    
    If input arguments are None, values are read from files.
        
    Return:
        Dataframe, where each row is a generation, composed by:
            bits of its best chromosome
            best score 
            diversity (obtained from the whole population, including best)
            time of life of the generation
    """
    if best_chromo is None:
        with open(MetricsPath + "Best_chromosome.pkl",'rb') as f:
            best_chromo = pickle.load(f)
        with open(MetricsPath + "Best_score.pkl",'rb') as f:
            best_score = pickle.load(f) 
        with open(MetricsPath + "Time.pkl", 'rb') as f:
            secondsGen = pickle.load(f)
    else:
        best_chromo = best_chromo
        best_score = best_score
        secondsGen = secondsGen
    
    Generation = []
    # Load last populationGen =============================================
    paths = [str(x) for x in Path(MetricsPath).glob('Population*End.txt')]
    Num_gens = len(paths)
    for GenPath in paths:
        with open(GenPath,'r') as f:
            temp = f.read()
        Generation.append(np.array(eval(temp.replace('.','.0,').replace('\n',','))))
        
    results_diversity = [] # store diversity of each generation
    for results_population in Generation: # shape=(NumbParents,NumbBitsDecimal)
        n_parents = len(results_population)
        distances = [] # store distance between each two chromosomes
        for i,j in itertools.combinations(range(n_parents),2):
            distances.append(distance.hamming(results_population[i], results_population[j]))
        results_diversity.append(np.mean(distances))
    
    # write chromosome as string
    str_best_chromo = [''.join([str(int(b)) for b in chromo]) for chromo in best_chromo]
    # Format time string
    format_secondsGen = []
    prev = 0.
    for s in secondsGen:
        days = int(strftime('%d', gmtime(s + prev))) - 1
        clock = strftime('%H:%M:%S', gmtime(s+prev))
        str_time = "{0} day(s), {1}" if days else "{1}"
        # prev = s + prev
        format_secondsGen.append(str_time.format(days,clock))
    
    # Each row of this dataframe represents a Generation:
    summary = pd.DataFrame({'Chromosome': str_best_chromo,
                            'AUC': best_score,
                            'Diversity': results_diversity,
                            'Time': format_secondsGen,
                            },
                           index=[f'Gen{i}' for i in range(Num_gens)])
    if verbose: print(summary)
    
    f = open(MetricsPath + "Diversity.txt", 'w')
    table_row = "{0:>6} {1:>22} {2:>10} {3:>10} {4:>22}"
    Diversity_tbl = [table_row.format('', 'Chromosome','AUC','Diversity','Time')] # headers
    for i in range(len(summary)):
        row = summary.iloc[i]
        Diversity_tbl.append(table_row.format(f'Gen{i}',row[0],f"{row[1]:.5f}",f"{row[2]:.5f}",row[3]))
    print("Diversity table:", *Diversity_tbl, sep='\n',file=f)
      
    return results_diversity, summary

###############################
#%% Hyperparameters for GA #
###############################
NumbParents:     int = 15 # Can't be lower than 2!! original=15
NumbBitsDecimal: int = 21 
CrossoverRate: float = 0.9 # Do not change
MutationRate:  float = 0.2 # Do not change
NumbGenerations: int = 10  # original=51
NumbGenerations: int = NumbGenerations+1 # Account for Generation 0
ElitsNumber:     int = 2

best_chromo: list = [] # best chromosome of each generation [GLOBAL VAR]
best_score:  list = [] # best score of best chromosome in best_chromo [GLOBAL VAR]
secondsGen:  list = [] #stores elapsed time for each generation

###############################################################################
#%% Starts algorithm: 
th = 0.
t0 = time()
t0log = datetime.now()
if fromCheckpoint == False:
    ## Clean all previous results manually ##
    with open(MetricsPath + "logGA.txt",'a') as f:
        to_write = "\n[log at {}]: New GA started.\n". \
            format(strftime('%H:%M:%S, %d/%b/%Y', localtime()))
        f.write(to_write)
        
    try:
        ## Generation 0 is randomly created $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        seconds = time()
        gen = 0
        population_firstgen = initilization_of_population(NumbParents, NumbBitsDecimal)
        scores_firstgen = np.zeros(shape=NumbParents)
        # Logs SEED generation
        with open(MetricsPath + "PopulationSeed.txt", 'w') as f:
            f.write(str(population_firstgen))
        
        # Log start of generation
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: Generation{} started.\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),gen)
            f.write(to_write)
            
        # Save Initial population of current generation
        with open(MetricsPath + f"Population{gen:0>2}Begin.txt", 'w') as f:
            f.write(str(population_firstgen))
        # Save Initial score of current generation
        with open(MetricsPath + f"Score{gen:0>2}Begin.txt", 'w') as f:
            f.write('['+'\n '.join(scores_firstgen.astype('str'))+']')
        
        print(f"\n\nNew generation: {gen}\n\n")
        scoresGen, population_nextgen = fitness_score(population_firstgen, gen)  
        
        # Save Final population of current generation
        with open(MetricsPath + f"Population{gen:0>2}End.txt", 'w') as f:
            f.write(str(population_nextgen))
        # Save Final score of current generation
        with open(MetricsPath + f"Score{gen:0>2}End.txt", 'w') as f:
            f.write('['+'\n '.join(scoresGen.astype('str'))+']')
        
        # Add first generation results to variables in memory as best
        best_score.append(scoresGen[0])
        best_chromo.append(population_nextgen[0])
        
        # Save all best chromosomes untill now
        with open(MetricsPath + "Best_score.pkl", 'wb') as f:
            pickle.dump(best_score, f)
        with open(MetricsPath + "Best_chromosome.pkl", 'wb') as f:
            pickle.dump(best_chromo, f)
            
        # Save elapsed time of training
        secondsGen.append(time() - seconds)
        with open(MetricsPath + "Time.pkl", 'wb') as f:
            pickle.dump(secondsGen, f)
            
        # Log End of generation
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: Generation{} ended successfully.\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),gen)
            f.write(to_write)
        
        # Log elapsed time
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA elapsed time -> {}\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),
                       str(datetime.now()-t0log))
                       # strftime('%H:%M:%S', gmtime(time()-t0)))
            f.write(to_write)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
        # Starts Natural Selection ================================================
        for gen_num in range(1, NumbGenerations):
            seconds = time()
            
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: Generation{} started.\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), gen_num)
                f.write(to_write)
            
            scoresGen, population_nextgen,\
                  best_score, best_chromo = generations(NumbParents, 
                                                        MutationRate, 
                                                        CrossoverRate, 
                                                        gen_num, 
                                                        population_nextgen, 
                                                        scoresGen, 
                                                        ElitsNumber,
                                                        best_score,
                                                        best_chromo)
            secondsGen.append(time() - seconds)
            with open(MetricsPath + "Time.pkl", 'wb') as f:
                pickle.dump(secondsGen, f)
                
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: Generation{} ended successfully.\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),gen_num)
                f.write(to_write)
            
            # Log elapsed time
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: GA elapsed time -> {}\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),
                           str(datetime.now()-t0log))
                           # strftime('%H:%M:%S', gmtime(time()-t0)))
                f.write(to_write)
        # =========================================================================
    except Exception as e:
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA raised an exception: '{}'. Saving cumulative time in th.pkl -> {}\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), e,
                       str(datetime.now()-t0log))
                       # strftime('%H:%M:%S', gmtime(time()-t0)))
            f.write(to_write)
        # Save t0 to be used when fromCheckpoint = True
        with open(MetricsPath + "th.pkl",'wb') as f:
            pickle.dump(datetime.now()-t0log,f)
        raise e("Failed during generation Fitness")
        
    except KeyboardInterrupt:
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA stopped by user.\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()))
            f.write(to_write)
        # Save t0 to be used when fromCheckpoint = True
        with open(MetricsPath + "th.pkl",'wb') as f:
            pickle.dump(datetime.now()-t0log,f)
        raise
    
else: #!!!
    ## Designed to continue optimization. If it was finished, this won't run
    # This will restart optimization from the beggining of last generation
    
    # Work-around *************************************************************
    # # "18:18:33, 30/Mar/2024" was taken from logGA.txt (FIRST TIME IT FAILED)
    # t0log_checkpoint = datetime.strptime("18:18:33, 30/Mar/2024", '%H:%M:%S, %d/%b/%Y')
    # th = datetime.now()-t0log_checkpoint
    
    # # "04:57:05, 22/Apr/2024" was taken from logGA.txt (SECOND TIME IT FAILED)
    # t0log_checkpoint = datetime.strptime("04:57:05, 22/Apr/2024", '%H:%M:%S, %d/%b/%Y')
    # th = datetime.now()-t0log_checkpoint
    
    # # "13:22:06, 12/May/2024" was taken from logGA.txt (THIRD TIME IT FAILED)
    # t0log_checkpoint = datetime.strptime("13:22:06, 12/May/2024", '%H:%M:%S, %d/%b/%Y')
    # th = datetime.now()-t0log_checkpoint
    
    # # "13:41:33, 30/May/2024" was taken from logGA.txt (STOPPED IT JUST IN CASE)
    # t0log_checkpoint = datetime.strptime("13:41:33, 30/May/2024", '%H:%M:%S, %d/%b/%Y')
    # th = datetime.now()-t0log_checkpoint
    
    # "07:28:01, 05/Jun/2024" was taken from logGA.txt (STOPPED DUE TO WIFI ISSUE)
    t0log_checkpoint = datetime.strptime("07:28:01, 05/Jun/2024", '%H:%M:%S, %d/%b/%Y')
    th = datetime.now()-t0log_checkpoint
    
    # # Load t0 to be used when fromCheckpoint = True
    # with open(MetricsPath + "th.pkl",'rb') as f:
    #     th = pickle.load(f)
    # *************************************************************************
    
    with open(MetricsPath + "logGA.txt",'a') as f:
        to_write = "\n[log at {}]: GA restarting from Checkpoint.\n". \
            format(strftime('%H:%M:%S, %d/%b/%Y', localtime()))
        f.write(to_write)
    
    try:
        # NOTE: PopulationXXEnd.txt is exactly equal to PopulationYYBegin.txt
        # NOTE: ScoreXXEnd.txt is exactly equal toScoreYYBegin.txt
        # So the method below is to restart at the end of the last population
        
        # Load last scoresGen =================================================
        paths = [str(x) for x in Path(MetricsPath).glob('Score*End.txt')]
        print("Reading Scores from files:", *paths,sep='\n')
        print()
        lastGenPath = paths[-1]
        
        with open(lastGenPath,'r') as f:
            temp = f.read()
        scoresGen = np.array(eval(temp.replace('\n',',')))
        nextGenerationS = len(paths)
        
        # Load last populationGen =============================================
        paths = [str(x) for x in Path(MetricsPath).glob('Population*End.txt')]
        print("Reading Populations from files:", *paths,sep='\n')
        lastGenPath = paths[-1]
        
        with open(lastGenPath,'r') as f:
            temp = f.read()
        population_nextgen = np.array(eval(temp.replace('.','.0,').replace('\n',',')))
        nextGenerationP = len(paths)
        
        # Load global arrays ==================================================
        with open(MetricsPath + "Best_chromosome.pkl", 'rb') as f:
            best_chromo = pickle.load(f)
        with open(MetricsPath + "Best_score.pkl", 'rb') as f:
            best_score = pickle.load(f)
        with open(MetricsPath + "Time.pkl", 'rb') as f:
            secondsGen = pickle.load(f)
        
        # Verify next generation ==============================================
        assert nextGenerationS == nextGenerationP, 'Saved chromosomes DO NOT match'
        assert len(best_chromo) == len(best_score), 'Saved chromosomes DO NOT match'
        assert len(best_chromo) == len(secondsGen), 'Saved chromosomes DO NOT match'
        assert nextGenerationS == len(best_chromo), 'Saved chromosomes DO NOT match'
        nextGeneration = nextGenerationS
        # Log success
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA restarted at beginning of Gen{}.\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), nextGeneration)
            f.write(to_write)
        
        if nextGeneration == NumbGenerations:
            print("\n  GA already run the desired number of generations  ")
        # Continue Natural Selection ================================================
        for gen_num in range(nextGeneration, NumbGenerations):
            seconds = time()
            
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: Generation{} started.\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()), gen_num)
                f.write(to_write)
            
            scoresGen, population_nextgen,\
                  best_score, best_chromo = generations(NumbParents, 
                                                        MutationRate, 
                                                        CrossoverRate, 
                                                        gen_num, 
                                                        population_nextgen, 
                                                        scoresGen, 
                                                        ElitsNumber,
                                                        best_score,
                                                        best_chromo)
            
            secondsGen.append(time() - seconds)
            with open(MetricsPath + "Time.pkl", 'wb') as f:
                pickle.dump(secondsGen, f)
                
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: Generation{} ended successfully.\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),gen_num)
                f.write(to_write)
                
            with open(MetricsPath + "logGA.txt",'a') as f:
                to_write = "[log at {}]: GA elapsed time -> {}\n". \
                    format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),
                           str(th + datetime.now()-t0log))
                           # strftime('%H:%M:%S', gmtime(th + time()-t0)))
                f.write(to_write)
        # =========================================================================
    except Exception as e:
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA raised an exception: '{}'. Saving cumulative time in th.pkl -> {}\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),e,
                       str(th + datetime.now()-t0log))
            f.write(to_write)
        # Save t0 to be used when fromCheckpoint = True
        with open(MetricsPath + "th.pkl",'wb') as f:
            pickle.dump(th + datetime.now()-t0log,f)
        raise e("Failed during generation Fitness")
        
    except KeyboardInterrupt:
        with open(MetricsPath + "logGA.txt",'a') as f:
            to_write = "[log at {}]: GA stopped by user.\n". \
                format(strftime('%H:%M:%S, %d/%b/%Y', localtime()))
            f.write(to_write)
        # Save t0 to be used when fromCheckpoint = True
        with open(MetricsPath + "th.pkl",'wb') as f:
            pickle.dump(th + datetime.now()-t0log,f)
        raise
        
# overallTime = th + time() - t0
overallTimelog = th + datetime.now()-t0log
print(f"Optimization time: {str(overallTimelog)}")

with open(MetricsPath + "logGA.txt",'a') as f:
    to_write = "[log at {}]: GA Finished! Total time spent -> {}\n". \
        format(strftime('%H:%M:%S, %d/%b/%Y', localtime()),
               str(overallTimelog))
    f.write(to_write)
    
#%% ================== Diversity of each generation =========================== 
"""
For each generation, I will calculate the hamming distance between their chromosomes
and then compute the average value.

If Diversity is very low, no more optimization is being done!
If Algorithm is working, diversity should decrease after every generation!
"""   
# results_diversity, summary = compute_diversity(best_chromo, best_score, 
#                                                secondsGen, verbose=True)

# From files: (usefull for running when data is not in memory anymore)
results_diversity, summary = compute_diversity(verbose=True)
