# -*- coding: utf-8 -*-
"""
Define how characteristics are encoded for the Genetic Algorithm
"""

def decode(crom, verbose, **locus):
    N = len(locus)
    decoded_chromosome = list(0 for _ in range(N)) # one decode per locus
    parameter = list(locus.keys())
    
    for i,k in enumerate(parameter):
        Nbits = len(list(locus[k].keys())[0])
        decoded_chromosome[i] = locus[k][crom[:Nbits]]
        crom = crom[Nbits:]
        if verbose: print(f"  {k}: {decoded_chromosome[i]}")
    
    return decoded_chromosome

def decode_chromosome(chromosome, verbose=True):
    """
    Receives an encoded chromosome and decodes it, returning its specifications
     chromosome: (array of float) the chromosome to be decoded
     verbose: (bool) whether to print the chromosome features
    return: (array) parameters of the decoded chromosome
    """
    # Convert chromosome to string        
    chromosome_str = ''.join([str(int(locus)) for locus in chromosome])
    
    # Decode the chromosome (21 bits)
    decoded_chromosome = \
    decode(chromosome_str, verbose,
           Bidirectional = {'0': '', '1': 'Bi'},
           NumberLSTMLayers = {'00': 1, '01': 2, '10': 3, '11': 4},
           PositionAttention = {'00': 'No', '01': 'pre', '10': 'post', '11': 'pre-post'},
           ModelDimension = {'00': 300, '01': 512, '10': 768, '11': 1024},
           ShapeOfProjection = {'00': 32, '01': 64, '10': 128, '11': 256},
           # NumberHeads = {}, # Calculated
           PercentageDropout = {'000': 0.05, '001': 0.10, '010': 0.15, '011': 0.20,
                                '100': 0.25, '101': 0.30, '110': 0.35, '111': 0.40},
           Decoder = {'00': 'Dense', '01': 'Pooling', '10': 'LSTM', '11': 'Bi-LSTM'},
           ShapeOfDenseLayer = {'00': 0.0, '01': 0.5, '10': 1.0, '11': 1.5},
           Activation = {'00': 'relu', '01': 'tanh', '10': 'leaky_relu', '11': 'elu'},
           Replicate = {'00': 0, '01': 1, '10': 2, '11': 3},
           Tokenizer = {'0': 'BlankSpace', '1': 'WordPiece'}
           )
        
    return decoded_chromosome
