# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:20:36 2023

@author: Admin

This file creates plots and analyzes Results of GA
"""

import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from time import time, strftime, gmtime, localtime
from scipy.spatial import distance
import matplotlib.pyplot as plt
from GeneticAlgorithm.encoding import decode_chromosome
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

results_folder = "AREC GA Results/Results/"
results_folder: str = "Article GA Results/Results/"


# Best_score.pkl have list of best AUC of each generation
with open(results_folder + "Best_score.pkl",'rb') as f:
    Best_score = pickle.load(f)
    
# Best_chromosome.pkl have list of best model coded in binary of each generation
with open(results_folder + "Best_chromosome.pkl",'rb') as f:
    Best_chromosome = pickle.load(f)
    
# Print parameters and score
besto = 5
for i in range(len(Best_chromosome)):
    print("Generation", i, "->", Best_chromosome[i])
    if i == besto:
        decoded_chromosome = decode_chromosome(Best_chromosome[i], verbose=True)
    else:
        _ = decode_chromosome(Best_chromosome[i], verbose=True)
    print("2-fold AUC:", Best_score[i],'\n')

vocab_size=11560
NumberClasses     =     3
NumberOfTimeSteps =     100
Optimizer         =     'rmsprop'
#                               
Bidirectional     =     decoded_chromosome[ 0] 
NumberLSTMLayers  =     decoded_chromosome[ 1]
PositionAttention =     decoded_chromosome[ 2]
ModelDimension    =     decoded_chromosome[ 3] 
ShapeOfProjection =     decoded_chromosome[ 4]
PercentageDropout =     decoded_chromosome[ 5]
Decoder           =     decoded_chromosome[ 6]
ShapeOfDenseLayer = int(decoded_chromosome[ 7]*ModelDimension) 
Activation        =     decoded_chromosome[ 8]
Replicate         =     decoded_chromosome[ 9]
Tokenizer         =     decoded_chromosome[10]
if ModelDimension == 300: # Particular case
    DxH ={32: 30, 64: 60, 128: 100, 256: 150}
    ShapeOfProjection = DxH[ShapeOfProjection]
NumberHeads       = int(ModelDimension/ShapeOfProjection)

# ==============================================================================
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional as Bi
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import TextVectorization
from keras_nlp.tokenizers import WordPieceTokenizer
# from Models.getMetrics import show_train_history
from main.selfAttention import (TokenAndPositionEmbedding,
                                                build_pos_matrix,
                                                MHAttention)

embedding_matrix = np.random.normal(size=(vocab_size,ModelDimension))
position_matrix = np.random.normal(size=(NumberOfTimeSteps,ModelDimension))

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
model_input = layers.Input(shape=(NumberOfTimeSteps,), name='Input_sequence')
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
# ==============================================================================

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
    MetricsPath = "./GeneticAlgorithm/Results/"
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
        
    return Generation
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

# # Diversity among all generations
# gen_diversity, summary = compute_diversity()
# _, ax = plt.subplots(figsize=(10,5),dpi=127)
# sns.lineplot(x=range(len(gen_diversity)),y=gen_diversity, ax=ax)
# plt.show()
    
# # Diversity between best chromosomes
# results_diversity = [] #store diversity of each generation
# j = len(Best_chromosome) -1
# for i in range(len(Best_chromosome)):
#     dist = distance.hamming(Best_chromosome[i], Best_chromosome[j])
#     results_diversity.append(dist)
    

# Fabio
# Load population of each generation
def loadPopulations():
    MetricsPath = "AREC GA Results/Results/"
    
    Populations = []
    # Load last populationGen =============================================
    paths = [str(x) for x in Path(MetricsPath).glob('Population*End.txt')]
    for GenPath in paths:
        with open(GenPath,'r') as f:
            temp = f.read()
        Populations.append(np.array(eval(temp.replace('.','.0,').replace('\n',','))))
        
    return np.asarray(Populations)

Populations = loadPopulations()
# 17 Populations:
#   10 parents (z)
#   16 length of each (L)

g = Populations.shape[0] # Number of generations
z = Populations.shape[1] # Number of chromosomes
L = Populations.shape[2] # Length of the chromosomes
number_of_combinations = z*(z-1)/2
Div = np.zeros(shape=g)
for gen in range(g):
    outter_sum = [] #store diversity of each generation
    for u in range(z-1):
        inner_sum = []
        for t in range(u+1, z):
            pu = Populations[gen,u]
            pt = Populations[gen,t]
            inner_sum.append( np.logical_xor(pu, pt).sum()/L )
        
        outter_sum.append( sum(inner_sum) )
    Div[gen] = sum(outter_sum) / number_of_combinations


#%% Plot results
GREEN: str = '#13501B'
ORANGE: str = '#C04F15'

fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=180)
fig.suptitle("Diversity vs. Score", fontsize=13)
ax2 = ax1.twinx()  
ax1.set_ylim(0.35, 0.55) # np.linspace(0.35, 0.55, 5)
ax2.set_ylim(91.5, 92.5) # np.linspace(91.5, 92.5, 5)

# X axis:
x = list(range(len(Best_chromosome)))
best_gen = 5 #np.argmax(Best_score)
# Diversity
ax1.plot(x, Div, '.-', markersize=15, color=GREEN)
ax1.axvline(x=best_gen, color='r', ls='--')

ax1.set_ylabel("Diversity", color=GREEN, fontsize=11)
# ax1.set_yticks(np.linspace(0.35, 0.5, 4))
ax1.tick_params(axis="y", labelcolor=GREEN)
ax1.set_xlabel('Generation', fontsize=11)
ax1.set_yticks(np.linspace(0.35, 0.55, 5))

ax1.grid(which='major', axis='both', alpha=0.7,
         color='k', linestyle='solid')
ax1.grid(which='minor', axis='y', alpha=0.7,
         color='gray', linestyle='dashed')
ax1.minorticks_on()

# AUC
ax2.plot(x, Best_score, '^--', markersize=8, color=ORANGE)
# ax2.axvline(x=best_gen, color='r', ls='--')

ax2.set_ylabel("AUC (%)", color=ORANGE, fontsize=11)
# ax2.set_yticks(np.linspace(91.8, 92.4, 4))
ax2.tick_params(axis="y", labelcolor=ORANGE)
# ax2.set_xlabel('Generation', fontsize=11)
ax2.set_yticks(np.linspace(91.5, 92.5, 5))

plt.show()


    
    
    
    
