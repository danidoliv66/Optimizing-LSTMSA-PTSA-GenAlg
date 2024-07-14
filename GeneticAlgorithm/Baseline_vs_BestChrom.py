# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:12:31 2024

@author: daniel

Description:

Find the best chromosome of each generation and plot them to see the convergence
of the architecture.

Each best chromosome is a binary string. If I represent that binary string as
its corresponding integer value (max = 2^21 -1) I could have a representation
of how distant are the chromosomes. 

Could this be equivalent to the diversity? Maybe, but diversity considers the
whole generation, while this method intends to understand what is the tendency 
of the best chromosome, individually.

"""
#%% Import libraries

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

#%% Constants
USE_RANDOM_DATA: bool = False
# RESULTS: str = "AREC GA Results/Results/"
RESULTS: str = "Article GA Results/Results/"

SEED:       str = "PopulationSeed.txt"
BEST_CHROM: str = "Best_chromosome.pkl"
BEST_SCORE: str = "Best_score.pkl"
SUMMARY:    str = "Diversity.txt"
LOG:        str = "logGA.txt"

POP_BEGIN: str = "Population{:0>2}Begin.txt"
SCO_BEGIN: str = "Score{:0>2}Begin.txt"
POP_END:   str = "Population{:0>2}End.txt"
SCO_END:   str = "Score{:0>2}End.txt"

NUMBER_GENERATIONS: int = 8
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

#%% Functions

def binary_to_int(binary_array):
    
    # Cast to integer values
    binary_array = np.asarray(binary_array, 'int')
    
    binary_str = ''.join(binary_array.astype(str))
    integer_value = int(binary_str, 2)
    return integer_value

#%% Get data

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
        
        
with open(RESULTS + BEST_CHROM, 'rb') as f:
    best_chromosomes = pickle.load(f)
    
# Cross check
for i in range(len(best_chromosomes)):
    print("1st chromosome matches best chromosome of gen{}: {}"
          .format(i, (best_chromosomes[i] == generations[i][0]).any() ))
    

#%% Plot
y = [binary_to_int(chrom) for chrom in best_chromosomes]
x = list(range(0, NUMBER_GENERATIONS))

fig, ax = plt.subplots(nrows=1, figsize=(7, 4), dpi=180)

ax.plot(x, y, '.-', linewidth=1.8, markersize=15, color=GREEN)

ax.set_ylabel("Avg. number of layers", fontsize=13)
ax.set_xlabel('Generation', fontsize=13)

ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=12)

ax.grid(which='major', axis='both', alpha=0.7,
        color='k', linestyle='solid')
ax.grid(which='minor', axis='y', alpha=0.7,
        color='gray', linestyle='dashed')
ax.minorticks_on()

fig.tight_layout(pad=2.0)
plt.show()




        
  
    
  
    
  
    
  