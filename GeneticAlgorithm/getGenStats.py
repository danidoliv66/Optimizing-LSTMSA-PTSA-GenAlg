# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:03:10 2024

@author: Daniel Parada
"""

"""
Get relevant stats of a Generic population.

Each population must have the same number of individuals (parameter)
Each individual must have the same number of bits (parameter)

The relevant stats are:
    - Number of LSTM layers
        > Number of MonoLSTM layers
        > Number of BiLSTM layers
        
    - Number of selfAttention layers
        > Number of PreAttention layers
        > Number of PostAttention layers
        
    - Dimensionality of the model (D)
    - Number of heads (H)
    - Dimensionality of projections (Dk)
    - Dropout percentage of the model (d)
    
    - Type of decoder (dec)
        > Dense
        > Pooling
        > LSTM
        > BiLSTM
    
    - Shape of dense layer (c)
    - Activation layer (act)
        
    - Tokenizer (T)
    - Vocabulary size (V)
    - Number of times the encoder is replicated (r)
    
    - Value of AUC
    - Ranking inside population
    - Time to train
    
"""
#%% Import libraries

import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from time import time, strftime, gmtime, localtime
from scipy.spatial import distance
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# import tensorflow as tf

# from tensorflow.keras import Model
# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Bidirectional as Bi
# from tensorflow.keras.layers import TextVectorization
# from keras_nlp.tokenizers import WordPieceTokenizer

# Local sources:
# from main.selfAttention import (TokenAndPositionEmbedding,
#                                                 build_pos_matrix,
#                                                 MHAttention)
from GeneticAlgorithm.encoding import decode_chromosome

# from ga_AREC.py:
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

#%% Constants
USE_RANDOM_DATA: bool = False
# RESULTS: str = "AREC GA Results/Results/"
# RESULTS: str = "Article GA Results/Results 8/"
# RESULTS: str = "Article GA Results/Results 9/"
# RESULTS: str = "Article GA Results/Results 10/"
RESULTS: str = "Article GA Results/Results 11/"

TIME:         str = "Time.pkl"
TIME_PER_POP: str = "TimePerPop.txt"

ACC: str = "ACC.txt"
AUC: str = "AUC.txt"
F1:  str = "F1.txt"
SEN: str = "Sen.txt"
SPE: str = "Spe.txt"

SEED:       str = "PopulationSeed.txt"
BEST_CHROM: str = "Best_chromosome.pkl"
BEST_SCORE: str = "Best_score.pkl"
SUMMARY:    str = "Diversity.txt"
LOG:        str = "logGA.txt"

POP_BEGIN: str = "Population{:0>2}Begin.txt"
SCO_BEGIN: str = "Score{:0>2}Begin.txt"
POP_END:   str = "Population{:0>2}End.txt"
SCO_END:   str = "Score{:0>2}End.txt"

TOP: int = 5

# Defult dictionary of Stats:
LSTM_UN: str = ''
LSTM_BI: str = 'Bi'
    
ATTENTION_NO:       str = "No"
ATTENTION_PRE:      str = "pre"
ATTENTION_POST:     str = "post"
ATTENTION_PRE_POST: str = "pre-post"
    
DECODER_DENSE:   str = "Dense"
DECODER_POOLING: str = "Pooling"
DECODER_LSTM:    str = "LSTM"
DECODER_BILSTM:  str = "Bi-LSTM"

ACT_RELU:       str = "relu"
ACT_TANH:       str = "tanh"
ACT_LEAKY_RELU: str = "leaky_relu"
ACT_ELU:        str = "elu"

TOKENIZER_BLANK: str = "BlankSpace"
TOKENIZER_PIECE: str = "WordPiece"

genStats_dict = {"Population ID": 0,
                 "Number of LSTM layers": [ ('LSTM',0), ('BiLSTM',0) ],
                 "Number of selfAttention layers": [ ('Pre',0), ('Post',0) ],
                 "Dimensionality of the model (D)": 0,
                 "Number of heads (H)": 0,
                 "Dimensionality of projections (Dk)": 0,
                 "Dropout percentage of the model (d)": 0.0,
                 "Type of decoder": '',
                 "Shape of dense layer (L)": 0,
                 "Activation layer": '',
                 "Tokenizer": '',
                 "Vocabulary size": 0,
                 "Number of replications": 0,
                 "Value of AUC": 0.0,
                 "Ranking inside population": 0,
                 "Time to train (h)": 0.0,
                 "Diversity": 0.0
                }

NUMBER_GENERATIONS: int = 11
NUMBER_CHROMOSOMES: int = 15
NUMBER_BITS: int = 21

LOC_BI:      int = 0
LOC_N_LSTM:  int = 1
LOC_POS_ATT: int = 2
LOC_DIM:     int = 3
LOC_PROJ:    int = 4
LOC_DROP:    int = 5
LOC_DEC:     int = 6
LOC_DENSE:   int = 7
LOC_ACT:     int = 8
LOC_REPL:    int = 9
LOC_TOK:     int = 10

GREEN: str = '#13501B'
ORANGE: str = '#C04F15'
LIGHT_GREEN: str = '#239133' 
LIGHT_ORANGE: str = '#EC8552'

#%% Auxiliar functions

def countRNN(genID: int, chromosome: list) -> list[tuple]:
    countLSTM: int = 0
    countBiLSTM: int = 0

    if chromosome[LOC_BI] == LSTM_UN:
        countLSTM = chromosome[LOC_N_LSTM] * (1 + chromosome[LOC_REPL])
    
    elif chromosome[LOC_BI] == LSTM_BI:
        countBiLSTM = chromosome[LOC_N_LSTM] * (1 + chromosome[LOC_REPL])
            
    if getDecoder(genID, chromosome) == DECODER_LSTM:
        countLSTM += 1
    elif getDecoder(genID, chromosome) == DECODER_BILSTM:
        countBiLSTM += 1
    
    return [('LSTM',countLSTM), ('BiLSTM',countBiLSTM)]

def countAttention(genID: int, chromosome: list) -> list[tuple]:
    countPre: int = 0
    countPost: int = 0

    if chromosome[LOC_POS_ATT] == ATTENTION_NO:
        return [('PreAtt',countPre), ('PostAtt',countPost)]
    
    elif chromosome[LOC_POS_ATT] == ATTENTION_PRE:
        countPre = 1 + chromosome[LOC_REPL]
        
    elif chromosome[LOC_POS_ATT] == ATTENTION_POST:
        countPost = 1 + chromosome[LOC_REPL]
        
    elif chromosome[LOC_POS_ATT] == ATTENTION_PRE_POST:
        countPre = 1 + chromosome[LOC_REPL]
        countPost = 1 + chromosome[LOC_REPL]
    
    return [('PreAtt',countPre), ('PostAtt',countPost)]

def getDimensions(genID: int, chromosome: list) -> tuple[int]:
    D: int   = 0 # Dimensionality of the model
    H: int   = 0 # Number of heads
    Dk: int  = 0 # Dimensionality of the projections
    Out: int = 0 # Dimensionality of classifier
    
    D = chromosome[LOC_DIM]
    Dk = chromosome[LOC_PROJ]
    Out = int(chromosome[LOC_DENSE]*D)
    
    if D == 300: # Particular case
        DxH ={32: 30, 64: 60, 128: 100, 256: 150}
        Dk = DxH[Dk]
    H = int(D/Dk)
    
    return D, H, Dk, Out

def getDecoder(genID: int, chromosome: list) -> str:
    decoderType: str = ''
    
    decoderType = chromosome[LOC_DEC]
    
    return decoderType

def getActivation(genID: int, chromosome: list) -> str:
    activationLayer: str = ''
    
    activationLayer = chromosome[LOC_ACT]
    
    return activationLayer

def getDropout(genID: int, chromosome: list) -> float:
    dropoutPercentage: float = 0.0
    # ...
    return dropoutPercentage

def getVocabulary(genID: int, chromosome: list) -> tuple[int, str]:
    vocabSize: int = 0
    tokenizer: str = ''
    
    # ...
    tokenizer = chromosome[LOC_TOK]
    
    return vocabSize, tokenizer

def getReplications(genID: int, chromosome: list) -> int:
    numberOfReplications: int = 0
    # ...
    return numberOfReplications

def getPerformancePop(genID: int, chromosome: list) -> tuple[float]:
    aucScore: float = 0.0
    trainHours: float = 0.0
    # ...
    return aucScore, trainHours

def getPerformanceGen(genID: int) -> pd.DataFrame:
    genTable = pd.DataFrame()
    # ...
    return genTable

def getDiversity(all_generations: list[np.ndarray]):
    g = len(all_generations) # Number of generations
    z = all_generations[0].shape[0] # Number of chromosomes
    L = all_generations[0].shape[1] # Length of the chromosomes
    number_of_combinations = z*(z-1)/2
    
    diversity = np.zeros(shape=g)
    for gen in range(g):
        outter_sum = [] # store diversity of each generation
        for u in range(z-1):
            inner_sum = []
            for t in range(u+1, z):
                pu = all_generations[gen][u]
                pt = all_generations[gen][t]
                inner_sum.append( np.logical_xor(pu, pt).sum()/L )
            
            outter_sum.append( sum(inner_sum) )
        diversity[gen] = sum(outter_sum) / number_of_combinations
        
    return diversity

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
        with open(RESULTS + BEST_CHROM,'rb') as f:
            best_chromo = pickle.load(f)
        with open(RESULTS + BEST_SCORE,'rb') as f:
            best_score = pickle.load(f) 
        with open(RESULTS + TIME, 'rb') as f:
            secondsGen = pickle.load(f)
    else:
        best_chromo = best_chromo
        best_score = best_score
        secondsGen = secondsGen
    
    Generation = []
    # Load last populationGen =============================================
    paths = [str(x) for x in Path(RESULTS).glob('Population*End.txt')]
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
    
    # f = open(MetricsPath + "Diversity.txt", 'w')
    table_row = "{0:>6} {1:>22} {2:>10} {3:>10} {4:>22}"
    Diversity_tbl = [table_row.format('', 'Chromosome','AUC','Diversity','Time')] # headers
    for i in range(len(summary)):
        row = summary.iloc[i]
        Diversity_tbl.append(table_row.format(f'Gen{i}',row[0],f"{row[1]:.5f}",f"{row[2]:.5f}",row[3]))
    print("Diversity table:", *Diversity_tbl, sep='\n')#,file=f)
      
    return results_diversity, summary

#%% Main
if __name__ == '__main__':
    
    if USE_RANDOM_DATA: 
        # Generate random generations
        generations = []
        for _ in range(NUMBER_GENERATIONS):
            generations.append(initilization_of_population(NUMBER_CHROMOSOMES, NUMBER_BITS))
            
        # Generate fake timing
        time_per_generation = []
        for _ in range(NUMBER_GENERATIONS):
            time_per_generation.append("{:>2} day(s), {:0>2}:{:0>2}:{:0>2}"\
                                       .format(np.random.randint(0, 7),
                                               np.random.randint(0, 24),
                                               np.random.randint(0, 60),
                                               np.random.randint(0, 60) ))
        # generate fake scores (...)
        
        
    else: # Use data from Results
        
        generations: list[np.ndarray] = [] # Read generations #!!!
        for gen in range(NUMBER_GENERATIONS):
            
            try:
                with open(RESULTS + POP_END.format(gen), 'r') as f:
                    tmp = ''.join(f.readlines())
                    tmp = tmp.replace('\n', ',')
                    tmp = tmp.replace('.', ',')
                    current_gen = np.array(eval(tmp),dtype='float')
                
                generations.append(current_gen)
            except:
                print("File", RESULTS + POP_END.format(gen), "not found")
        
        
        time_per_generation: list[tuple] = [] # Get timing #!!!
        # Each tuple is (<days>, <hours>, <minutes>)
        
        with open(RESULTS + TIME, 'rb') as f:
            tmp = pickle.load(f)
        
        for i in range(len(tmp)):
            total_seconds = tmp[i]
            days = total_seconds / (60*60*24)
            hours = (days % int(days)) * 24
            minutes = (hours % int(hours)) * 60
            timing = (int(days), int(hours), round(minutes))
            
            time_per_generation.append(timing)
            
        # Load time per architecture from TimePerPop.txt 
        avg_time_per_pop_in_generation: list[float] = []
        # Stores the time in hours
        
        with open(RESULTS + TIME_PER_POP, 'r') as f:
            reg = list(filter(lambda x: x != '\n', f.readlines()))
        
        k = 0 # line number
        sep = " --> "
        pattern = "[Gen{}|Chrom{}] fold{}"
        
        for i in range(NUMBER_GENERATIONS):
            
            sum_timing = 0
            
            for j in range(1, NUMBER_CHROMOSOMES+1):
                
                line = reg[k].strip()
                header, data = line.split(sep)
                data = data.split(':')
                
                if header == pattern.format(i, j, 0):
                    time_fold0 = int(data[0]) + (int(data[1]) / 60) + (int(data[2]) / 3600)
                
                k += 1
                
                line = reg[k].strip()
                header, data = line.split(sep)
                data = data.split(':')
                    
                if header == pattern.format(i, j, 1):
                    time_fold1 = int(data[0]) + (int(data[1]) / 60) + (int(data[2]) / 3600)
                    
                k += 1
                
                timing = [time_fold0, time_fold1]
                sum_timing += sum(timing)
                
            avg_time_per_pop_in_generation.append(sum_timing/NUMBER_CHROMOSOMES)
                
                
        
        max_score_per_generation: list = [] # Get scores #!!!
        avg_score_per_generation: list = []
        for gen in range(NUMBER_GENERATIONS):
            
            try:
                with open(RESULTS + SCO_END.format(gen), 'r') as f:
                    tmp = ''.join(f.readlines())
                    tmp = tmp.replace('\n', ',')
                    current_scores = eval(tmp)
                
                max_score_per_generation.append(np.max(current_scores))
                avg_score_per_generation.append(np.mean(current_scores))
            except:
                print("File", RESULTS + SCO_END.format(gen), "not found")
                
        # with open(RESULTS + BEST_SCORE, 'rb') as f:
        #     best_scores = pickle.load(f)
            
        # # Cross check
        # for i in range(len(best_scores)):
        #     print("1st chromosome matches best chromosome of gen{}: {}"
        #           .format(i, (best_scores[i] == max_score_per_generation[i]) ))
        
        # ============================================================================
        # Fix scores!
        # NOTE: the values were randomized for the first 2 generations :/
        # I can get some insights from the redundant files
        max_score_per_generation[0] = 92.005
        avg_score_per_generation[0] = 91.62386666666667
        """
        [New generation: 0]
            Population 01: 91.853 ± 0.1624
            Population 02: 91.986 ± 0.1116
            Population 03: 91.283 ± 0.1749
            Population 04: 91.185 ± 0.0325
            Population 05: 91.631 ± 0.1369
            Population 06: 91.687 ± 0.0342
            Population 07: 91.704 ± 0.1362
            Population 08: 91.745 ± 0.1803
            Population 09: 92.054 ± 0.0761
            Population 10: 91.322 ± 0.0265
            Population 11: 90.950 ± 0.1156
            Population 12: 91.331 ± 0.3602
            Population 13: 91.659 ± 0.2202
            Population 14: 91.963 ± 0.0413
            Population 15: 92.005 ± 0.0510
        """
        max_score_per_generation[1] = 92.042
        avg_score_per_generation[1] = 91.62753333333335
        """
        [New generation: 1]
            Population 01: 91.609 ± 0.0432
            Population 02: 91.870 ± 0.0269
            Population 03: 91.765 ± 0.0610
            Population 04: 91.525 ± 0.2061
            Population 05: 92.068 ± 0.0807
            Population 06: 91.773 ± 0.1206
            Population 07: 91.852 ± 0.1023
            Population 08: 92.064 ± 0.0069
            Population 09: 91.863 ± 0.0147
            Population 10: 91.159 ± 0.4741
            Population 11: 90.734 ± 0.1523
            Population 12: 91.945 ± 0.0556
            Population 13: 92.006 ± 0.1159
            Population 14: 90.880 ± 0.1731
            Population 15: 91.300 ± 0.1857
        """
        
        # ============================================================================
    
    #%% GET ATTENTION LAYERS DATA ################################################ #!!!
    # Store count of all generations
    all_gen_att_count = NUMBER_GENERATIONS*[0]
    all_gen_att_pre  =  NUMBER_GENERATIONS*[0]
    all_gen_att_post =  NUMBER_GENERATIONS*[0]
    # Store top5 count of all generations
    top5_gen_att_count = NUMBER_GENERATIONS*[0]
    top5_gen_att_pre  =  NUMBER_GENERATIONS*[0]
    top5_gen_att_post =  NUMBER_GENERATIONS*[0]
    
    for gen in range(NUMBER_GENERATIONS):
    
        population = generations[gen]
        print(f"Generation {gen}", population, sep='\n')
    
        gen_att_count = 0
        gen_att_pre  =  0
        gen_att_post =  0
        
        for i,chrom in enumerate(population):
            
            dec_chrom = decode_chromosome(chrom)
            print()
            
            att_count = countAttention(gen, dec_chrom)
            print(att_count)
            print()
            
            if i == TOP:
                top5_gen_att_count[gen] = gen_att_count
                top5_gen_att_pre[gen]  =  gen_att_pre
                top5_gen_att_post[gen] =  gen_att_post
            
            gen_att_count += att_count[0][1] + att_count[1][1]
            gen_att_pre += att_count[0][1]
            gen_att_post += att_count[1][1]
            
        all_gen_att_count[gen] += gen_att_count
        all_gen_att_pre[gen] += gen_att_pre
        all_gen_att_post[gen] += gen_att_post
        
    # Plot Average number of Attention layers per generation
    y1     = [n/NUMBER_CHROMOSOMES for n in all_gen_att_count]
    y1_top = [n/TOP for n in top5_gen_att_count]
    y2     = [n/NUMBER_CHROMOSOMES for n in all_gen_att_pre]
    y2_top = [n/TOP for n in top5_gen_att_pre]
    y3     = [n/NUMBER_CHROMOSOMES for n in all_gen_att_post]
    y3_top = [n/TOP for n in top5_gen_att_post]
    
    Y = [(y1, y1_top), (y2, y2_top), (y3, y3_top)]
    x = list(range(0, NUMBER_GENERATIONS))

    fig, ax = plt.subplots(nrows=3, figsize=(8, 11.5), dpi=300)
    title = ["All Attention layers", "Pre Attention layers",
             "Post Attention layers"]
    
    for i in range(len(ax)):
        ax[i].plot(x, Y[i][0], '.-', linewidth=1.8, markersize=15, color=GREEN)
        ax[i].plot(x, Y[i][1], '^--', markersize=8, color=ORANGE)
        
        ax[i].set_title(title[i], fontsize=16)
        ax[i].set_ylabel("Avg. number of layers", fontsize=16)
        if i == 2:
            ax[i].set_xlabel('Generation', fontsize=16)
        
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(x, fontsize=15)
        if i == 0:
            ax[i].set_yticklabels([f'{n:.1f}' for n in ax[i].get_yticks()], fontsize=15)
        elif i == 1:
            ax[i].set_yticks(np.linspace(0.0, 2.0, 5))
            ax[i].set_yticklabels([f'{n:.1f}' for n in ax[i].get_yticks()], fontsize=15)
        elif i == 2:
            ax[i].set_yticklabels([f'{n:.1f}' for n in ax[i].get_yticks()], fontsize=15)
        
        # ax[i].legend(['All chromosomes', 'Top5 chromosomes'], 
        #            fontsize=11, loc='best')
        ax[i].grid(which='major', axis='both', alpha=0.7,
                   color='k', linestyle='solid')
        ax[i].grid(which='minor', axis='y', alpha=0.7,
                   color='gray', linestyle='dashed')
        ax[i].minorticks_on()
        
    ax[0].legend(['All chromosomes', f'Top{TOP} chromosomes'], 
               fontsize=15, loc='best')
    
    fig.tight_layout(pad=2.0)
    
    plt.savefig("Article GA Results/images/attention.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/attention.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################
    
    #%% GET LSTM LAYERS DATA ##################################################### #!!!
    # Store count of all generations
    all_gen_lstm_count = NUMBER_GENERATIONS*[0]
    all_gen_unlstm = NUMBER_GENERATIONS*[0]
    all_gen_bilstm = NUMBER_GENERATIONS*[0]
    # Store top5 count of all generations
    top5_gen_lstm_count = NUMBER_GENERATIONS*[0]
    top5_gen_unlstm = NUMBER_GENERATIONS*[0]
    top5_gen_bilstm = NUMBER_GENERATIONS*[0]

    for gen in range(NUMBER_GENERATIONS):
    
        population = generations[gen]
        print(f"Generation {gen}", population, sep='\n')
        
        gen_lstm_count   = 0
        gen_unlstm_count = 0
        gen_bilstm_count = 0
        
        for i,chrom in enumerate(population):
            
            dec_chrom = decode_chromosome(chrom)
            print()
            
            lstm_count = countRNN(gen, dec_chrom)
            print(lstm_count)
            print()
            
            if i == TOP:
                top5_gen_lstm_count[gen] = gen_lstm_count
                top5_gen_unlstm[gen]  =  gen_unlstm_count
                top5_gen_bilstm[gen] =  gen_bilstm_count
            
            gen_lstm_count += lstm_count[0][1] + lstm_count[1][1]
            gen_unlstm_count += lstm_count[0][1]
            gen_bilstm_count += lstm_count[1][1]
            
        all_gen_lstm_count[gen] += gen_lstm_count
        all_gen_unlstm[gen] += gen_unlstm_count
        all_gen_bilstm[gen] += gen_bilstm_count
        
    # Plot Average number of Attention layers per generation
    y1     = [n/NUMBER_CHROMOSOMES for n in all_gen_lstm_count]
    y1_top = [n/TOP for n in top5_gen_lstm_count]
    y2     = [n/NUMBER_CHROMOSOMES for n in all_gen_unlstm]
    y2_top = [n/TOP for n in top5_gen_unlstm]
    y3     = [n/NUMBER_CHROMOSOMES for n in all_gen_bilstm]
    y3_top = [n/TOP for n in top5_gen_bilstm]
    
    Y = [(y1, y1_top), (y2, y2_top), (y3, y3_top)]
    x = list(range(0, NUMBER_GENERATIONS))

    fig, ax = plt.subplots(nrows=3, figsize=(8, 11.5), dpi=300)
    title = ["All LSTM layers", "Unidirectional LSTM layers",
             "Bidirectional LSTM layers"]
    
    for i in range(len(ax)):
        ax[i].plot(x, Y[i][0], '.-', linewidth=1.8, markersize=15, color=GREEN)
        ax[i].plot(x, Y[i][1], '^--', markersize=8, color=ORANGE)
        
        ax[i].set_title(title[i], fontsize=16)
        ax[i].set_ylabel("Avg. number of layers", fontsize=16)
        if i == 2:
            ax[i].set_xlabel('Generation', fontsize=16)
        
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(x, fontsize=15)
        ax[i].set_yticklabels([f'{n:.0f}' for n in ax[i].get_yticks()], fontsize=15)
        
        # ax[i].legend(['All chromosomes', f'Top{TOP} chromosomes'], 
        #            fontsize=11, loc='best')
        ax[i].grid(which='major', axis='both', alpha=0.7,
                   color='k', linestyle='solid')
        ax[i].grid(which='minor', axis='y', alpha=0.7,
                   color='gray', linestyle='dashed')
        ax[i].minorticks_on()
    
    fig.tight_layout(pad=2.0)
    
    plt.savefig("Article GA Results/images/lstm.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/lstm.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################
    
    #%% GET DIMENSIONALITY DATA ################################################## #!!!
    from collections import Counter
    
    # Count only from top 5 chromosomes
    count_D = Counter()
    count_H = Counter()
    count_Dk = Counter()
    count_Out = Counter()
    
    for gen in range(NUMBER_GENERATIONS-4, NUMBER_GENERATIONS):
    
        population = generations[gen]
        print(f"Generation {gen}", population, sep='\n')
        
        # gen_D_count = 0
        # gen_H_count = 0
        # gen_Dk_count = 0
        # gen_Out_count = 0
        
        for i,chrom in enumerate(population):
            dec_chrom = decode_chromosome(chrom)
            print()
            
            D, H, Dk, Out = getDimensions(gen, dec_chrom)
            print(f"Model dimension: {D}\nHeads: {H}\nProjections: {Dk}\nClassifier: {Out if Out else 'No'}")
            print()
            
            count_D[D] += 1
            count_H[H] += 1
            count_Dk[Dk] += 1
            count_Out[Out] += 1
            
            if i == TOP-1: break
    
    # Plot prefered dimensionality of top5 chromosomes of each generation
    
    fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    # fig.suptitle(f"Preferred dimensionality of top{TOP} chromosomes\n(last 3 generations)", fontsize=14)
    
    x = sorted(count_D.keys())
    y = [count_D[val] for val in x]
    x_axis = range(len(x))
    x_labels = [str(n) for n in x]
    ax1.bar(x_axis, y, 0.5, color=GREEN)
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(x_labels, fontsize=16)
    ax1.set_yticklabels([f'{n:.0f}' for n in ax1.get_yticks()], fontsize=16)
    ax1.set_title("Model dimension", fontsize=16)
    
    ax1.grid(which='major', axis='y', alpha=0.7,
             color='k', linestyle='solid')
    ax1.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax1.minorticks_on()
    
    x = sorted(count_H.keys())
    y = [count_H[val] for val in x]
    x_axis = range(len(x))
    x_labels = [str(n) for n in x]
    ax2.bar(x_axis, y, 0.5, color=ORANGE)
    ax2.set_xticks(x_axis)
    ax2.set_xticklabels(x_labels, fontsize=16)
    ax2.set_yticklabels([f'{n:.0f}' for n in ax2.get_yticks()], fontsize=16)
    ax2.set_title("Number of Attention heads", fontsize=16)
    
    ax2.grid(which='major', axis='y', alpha=0.7,
             color='k', linestyle='solid')
    ax2.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax2.minorticks_on()
    
    x = sorted(count_Dk.keys())
    y = [count_Dk[val] for val in x]
    x_axis = range(len(x))
    x_labels = [str(n) for n in x]
    ax3.bar(x_axis, y, 0.5, color=LIGHT_GREEN)
    ax3.set_xticks(x_axis)
    ax3.set_xticklabels(x_labels, fontsize=16)
    ax3.set_yticklabels([f'{n:.0f}' for n in ax3.get_yticks()], fontsize=16)
    ax3.set_title("Heads projection", fontsize=16)
    
    ax3.grid(which='major', axis='y', alpha=0.7,
             color='k', linestyle='solid')
    ax3.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax3.minorticks_on()
    
    x = sorted(count_Out.keys())
    y = [count_Out[val] for val in x]
    x_axis = range(len(x))
    x_labels = [str(n) for n in x]
    ax4.bar(x_axis, y, 0.5, color=LIGHT_ORANGE)
    ax4.set_xticks(x_axis)
    ax4.set_xticklabels(['No'] + x_labels[1:], fontsize=16)
    ax4.set_yticklabels([f'{n:.0f}' for n in ax4.get_yticks()], fontsize=16)
    ax4.set_title("Classifier dimensionality", fontsize=16)
    
    ax4.grid(which='major', axis='y', alpha=0.7,
             color='k', linestyle='solid')
    ax4.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax4.minorticks_on()
    
    plt.tight_layout()
    plt.savefig("Article GA Results/images/dimension.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/dimension.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################

    #%% GET TOKENIZER DATA ####################################################### #!!!
    # Store count of all generations
    all_blankspace_count = NUMBER_GENERATIONS*[0]
    all_wordpiece_count = NUMBER_GENERATIONS*[0]
    
    for gen in range(NUMBER_GENERATIONS):
    
        population = generations[gen]
        print(f"Generation {gen}", population, sep='\n')
        
        gen_blankspace_count = 0
        gen_wordpiece_count = 0
        
        for i,chrom in enumerate(population):
            dec_chrom = decode_chromosome(chrom)
            print()
            
            _, tokenizer_type = getVocabulary(gen, dec_chrom)
            print(tokenizer_type)
            print()
            
            gen_blankspace_count += tokenizer_type == TOKENIZER_BLANK
            gen_wordpiece_count += tokenizer_type == TOKENIZER_PIECE
            
        all_blankspace_count[gen] += gen_blankspace_count
        all_wordpiece_count[gen] += gen_wordpiece_count
        
    x = list(range(0, NUMBER_GENERATIONS))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    y_offset = np.zeros(NUMBER_GENERATIONS)
    ax.bar(x, all_blankspace_count, 0.6, label='Blankspace', bottom=y_offset,
           color=LIGHT_ORANGE)
    y_offset += all_blankspace_count
    ax.bar(x, all_wordpiece_count, 0.6, label='Wordpiece', bottom=y_offset,
           color=LIGHT_GREEN)
    
    ax.set_yticks(list(range(NUMBER_CHROMOSOMES+1)), fontsize=12)
    ax.set_xticks(x, fontsize=12)
    
    ax.set_ylabel("Number of chromosomes", fontsize=13)
    ax.set_xlabel('Generation', fontsize=13)
    
    ax.legend(fontsize=11, loc='lower right', framealpha=0.8)
    ax.grid(which='major', axis='y',
            color='k', linestyle='solid')
    
    plt.savefig("Article GA Results/images/tokenizer.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/tokenizer.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    
    ############################################################################
    
    #%% GET ACTIVATION LAYER DATA ################################################ #!!!
    # Store count of all generations
    all_relu_count = NUMBER_GENERATIONS*[0]
    all_tanh_count = NUMBER_GENERATIONS*[0]
    all_leaky_relu_count = NUMBER_GENERATIONS*[0]
    all_elu_count = NUMBER_GENERATIONS*[0]
    
    for gen in range(NUMBER_GENERATIONS):
    
        population = generations[gen]
        print(f"Generation {gen}", population, sep='\n')
        
        gen_relu_count = 0
        gen_tanh_count = 0
        gen_leaky_relu_count = 0
        gen_elu_count = 0
        
        for i,chrom in enumerate(population):
            dec_chrom = decode_chromosome(chrom)
            print()
            
            activation = getActivation(gen, dec_chrom)
            print(activation)
            print()
            
            gen_relu_count += activation == ACT_RELU
            gen_tanh_count += activation == ACT_TANH
            gen_leaky_relu_count += activation == ACT_LEAKY_RELU
            gen_elu_count += activation == ACT_ELU
            
        all_relu_count[gen] += gen_relu_count
        all_tanh_count[gen] += gen_tanh_count
        all_leaky_relu_count[gen] += gen_leaky_relu_count
        all_elu_count[gen] += gen_elu_count
        
    x = list(range(0, NUMBER_GENERATIONS))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    y_offset = np.zeros(NUMBER_GENERATIONS)
    ax.bar(x, all_relu_count, 0.6, label='ReLU', bottom=y_offset,
           color=GREEN)
    y_offset += all_relu_count
    ax.bar(x, all_tanh_count, 0.6, label='Tanh', bottom=y_offset,
           color=LIGHT_GREEN)
    y_offset += all_tanh_count
    ax.bar(x, all_leaky_relu_count, 0.6, label='Leaky ReLU', bottom=y_offset,
           color=ORANGE)
    y_offset += all_leaky_relu_count
    ax.bar(x, all_elu_count, 0.6, label='eLU', bottom=y_offset,
           color=LIGHT_ORANGE)
    y_offset += all_elu_count
    
    ax.set_yticks(list(range(NUMBER_CHROMOSOMES+1)), fontsize=12)
    ax.set_xticks(x, fontsize=12)
    
    ax.set_ylabel("Number of chromosomes", fontsize=13)
    ax.set_xlabel('Generation', fontsize=13)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.8)
    ax.grid(which='major', axis='y',
            color='k', linestyle='solid')
    
    plt.savefig("Article GA Results/images/activation.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/activation.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################

    #%% GET AVERAGE TIME DATA #################################################### #!!!
  
    # Plot time per generation
    y1_days = [t[0] + t[1]/24 + t[2]/(60*24) for t in time_per_generation]
    y2_hours = avg_time_per_pop_in_generation
    x = list(range(0, NUMBER_GENERATIONS))

    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=300)
    # fig.suptitle("Duration of optimization", fontsize=13)
    fig.suptitle(" ", fontsize=13)
    ax2 = ax1.twinx()  
    ax1.set_ylim(2, 10) # np.linspace(2.5, 6.5, 5)
    ax2.set_ylim(3, 13) # np.linspace(3, 13, 5)
    
    ax1.plot(x, y1_days, '.-', markersize=15, color=GREEN)
    
    # ax1.set_title("Dimensionality of top5 chromosomes per generation", fontsize=14)
    ax1.set_ylabel("Total duration (days)", color=GREEN, fontsize=15)
    ax1.set_xlabel('Generation', fontsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, fontsize=15)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax1.set_yticks(np.linspace(2, 10, 5), fontsize=15)
    ax1.set_yticklabels([f'{n:.0f}' for n in ax1.get_yticks()], fontsize=15)
    
    # ax.legend(['Model Dimension', 'Projection', 'Classification layer'], 
    #           fontsize=8, loc='best')
    ax1.grid(which='major', axis='both', alpha=0.7,
             color='k', linestyle='solid')
    ax1.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax1.minorticks_on()
    
    
    ax2.plot(x, y2_hours, '^--', markersize=8, color=ORANGE)
    
    ax2.set_ylabel("Avg. duration\nper chromosome (hours)", color=ORANGE, fontsize=15)
    ax2.set_xticks(x)
    # ax2.set_xticklabels(x, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_yticks(np.linspace(3, 13, 5))
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=15)
    
    plt.tight_layout()#pad=1.08, h_pad=None, w_pad=None, rect=None)
    
    plt.savefig("Article GA Results/images/time.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/time.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################
    
    #%% GET Diversity ######################################################### #!!!
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=300)
    # fig.suptitle("Diversity vs. Score", fontsize=13)
    fig.suptitle(" ", fontsize=13)
    ax2 = ax1.twinx()  
    ax1.set_ylim(0.3, 0.6) # np.linspace(0.35, 0.55, 5)
    ax2.set_ylim(91.8, 92.4) # np.linspace(91.5, 92.5, 5)

    # X axis:
    x = list(range(0, NUMBER_GENERATIONS))
    best_gen = 7 # np.argmax(Best_score)
    y1 = getDiversity(generations)
    y2 = max_score_per_generation
    
    # Diversity
    if best_gen > 0:
        ax1.axvline(x=best_gen, color='r', ls='--')
    ax1.plot(x, y1, '.-', markersize=15, color=GREEN)

    ax1.set_ylabel("Diversity", color=GREEN, fontsize=15)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax1.set_xlabel('Generation', fontsize=15)
    ax1.set_yticks(np.linspace(0.3, 0.6, 7))
    ax1.set_yticklabels([f'{n:.2f}' for n in ax1.get_yticks()], fontsize=15)

    ax1.grid(which='major', axis='both', alpha=0.7,
             color='k', linestyle='solid')
    ax1.grid(which='minor', axis='y', alpha=0.7,
             color='gray', linestyle='dashed')
    ax1.minorticks_on()

    # AUC
    ax2.plot(x, y2, '^--', markersize=8, color=ORANGE)
    ax2.set_ylabel("AUC (%)", color=ORANGE, fontsize=15)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_yticks(np.linspace(91.8, 92.4, 7))
    ax2.set_yticklabels([f'{n:.1f}' for n in ax2.get_yticks()], fontsize=15)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(map(str, range(0, NUMBER_GENERATIONS))), fontsize=15)
    
    
    plt.tight_layout()#pad=1.08, h_pad=None, w_pad=None, rect=None)
    
    plt.savefig("Article GA Results/images/diversity.png", dpi=300, transparent=False, format='png')
    plt.savefig("Article GA Results/images/diversity.svg", dpi=300, transparent=False, format='svg')
    plt.show()
    ############################################################################
    
    #%% BEST CHROMOSOME ########################################################
    for i in range(NUMBER_GENERATIONS):
        print("\nGeneration", i)
        decode_chromosome(generations[i][0])
        print()
        decode_chromosome(generations[i][1])
        
    ############################################################################
    
    #%% SUMMARY #
    # Using the dictionary genStats_dict
    


from datetime import datetime 
        
         
time0 = datetime.strptime("17:06:23, 06/Apr/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("18:18:33, 30/Mar/2024", '%H:%M:%S, %d/%b/%Y')
        
time1 = datetime.strptime("10:47:55, 14/Apr/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("17:06:23, 06/Apr/2024", '%H:%M:%S, %d/%b/%Y')
        
time2 = datetime.strptime("04:57:05, 22/Apr/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("20:26:21, 15/Apr/2024", '%H:%M:%S, %d/%b/%Y')
        
time3 = datetime.strptime("04:34:45, 07/May/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("15:59:42, 02/May/2024", '%H:%M:%S, %d/%b/%Y')
        
time4 = datetime.strptime("13:22:06, 12/May/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("04:34:45, 07/May/2024", '%H:%M:%S, %d/%b/%Y')
        
time5 = datetime.strptime("04:45:35, 24/May/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("12:16:59, 18/May/2024", '%H:%M:%S, %d/%b/%Y')
        
time6 = datetime.strptime("13:41:33, 30/May/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("04:45:35, 24/May/2024", '%H:%M:%S, %d/%b/%Y') 
        
time7 = datetime.strptime("07:28:01, 05/Jun/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("15:35:25, 30/May/2024", '%H:%M:%S, %d/%b/%Y') 
       
time8 = datetime.strptime("14:45:05, 12/Jun/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("19:13:10, 06/Jun/2024", '%H:%M:%S, %d/%b/%Y') 
        
time9 = datetime.strptime("17:15:55, 18/Jun/2024", '%H:%M:%S, %d/%b/%Y') - \
        datetime.strptime("15:45:43, 12/Jun/2024", '%H:%M:%S, %d/%b/%Y') 
        
time10 = datetime.strptime("10:30:43, 27/Jun/2024", '%H:%M:%S, %d/%b/%Y') - \
         datetime.strptime("20:41:54, 18/Jun/2024", '%H:%M:%S, %d/%b/%Y') 

for i in range(NUMBER_GENERATIONS):
    et = eval(f'time{i}')
    if i == 0: total_time = et
    else: total_time += et
    print(f"Generation {i}: {et}")

print("Total time:", total_time)


#%%

# Besto: #!!!
np.array([1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.])

# with open(RESULTS + "PopulationParentsToFit.pkl", 'rb') as f:
#     actual_pop = np.array([])
#     prev_actual_pop = np.array([])
#     while(True):
#         print((prev_actual_pop == actual_pop).any())
#         try: 
#             actual_pop = pickle.load(f)
#         except: break
#         prev_actual_pop = actual_pop.copy()
        
        
#%%
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
        with open(RESULTS + BEST_CHROM,'rb') as f:
            best_chromo = pickle.load(f)
        with open(RESULTS + BEST_SCORE,'rb') as f:
            best_score = pickle.load(f) 
        with open(RESULTS + TIME, 'rb') as f:
            secondsGen = pickle.load(f)
    else:
        best_chromo = best_chromo
        best_score = best_score
        secondsGen = secondsGen
    
    Generation = []
    # Load last populationGen =============================================
    paths = [str(x) for x in Path(RESULTS).glob('Population*End.txt')]
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
    
    f = open(RESULTS + "Diversity.txt", 'w')
    table_row = "{0:>6} {1:>22} {2:>10} {3:>10} {4:>22}"
    Diversity_tbl = [table_row.format('', 'Chromosome','AUC','Diversity','Time')] # headers
    for i in range(len(summary)):
        row = summary.iloc[i]
        Diversity_tbl.append(table_row.format(f'Gen{i}',row[0],f"{row[1]:.5f}",f"{row[2]:.5f}",row[3]))
    print("Diversity table:", *Diversity_tbl, sep='\n',file=f)
      
    return results_diversity, summary



