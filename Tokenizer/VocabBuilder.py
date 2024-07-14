# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:43:29 2023

@author: daniel
"""

#%% Imports
import os
import numpy as np
import pandas as pd
from pathlib import Path
from time import time, strftime, gmtime
from tensorflow.keras.layers import TextVectorization
from keras_nlp.tokenizers import WordPieceTokenizer
from tokenizers import BertWordPieceTokenizer
from gensim.models.word2vec import Word2Vec

#%% Functions
def writeCorpus(lines, path='tmp/Corpus.txt'):
    # Write each document (review) in one separated line
    with open(path, 'w', encoding="utf-8") as f:
        for i,line in enumerate(lines):
            f.write(line)
            f.write('\n')
        else: print(i+1, "lines written into file")
    return path

def WordPieceVocabulary(corpus: pd.Series,
                        vector_size: int,
                        maxlen: int,
                        vocab_size: int = None) -> list[str]:
    temp_path: str = 'tmp'

    text_data = []
    file_count = 0
    for sample in corpus:
        text_data.append(sample)
        if len(text_data) == 25_000:
            with open(f'{temp_path}/text_{file_count:0>2}.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    if len(text_data):
        with open(f'{temp_path}/text_{file_count:0>2}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        
    # Check stored files:
    paths = [str(x) for x in Path(f'{temp_path}').glob('text_*.txt')]
    print(*paths,sep='\n')
    
    tokenizer = BertWordPieceTokenizer(clean_text=False,
                                       handle_chinese_chars=True,
                                       strip_accents=False,
                                       lowercase=False)
    tokenizer.train(files=paths, 
                    vocab_size=vocab_size or 30_000, 
                    min_frequency=3,
                    limit_alphabet=1000, 
                    wordpieces_prefix='##',
                    special_tokens=['[PAD]', '[UNK]']
                    )
    tokenizer.save_model(temp_path)
    
    with open(temp_path + '/vocab.txt', 'r',encoding='utf-8') as f:
        vocab_list = list(map(str.strip,f.readlines()))

    # Word2Vec model:
    wv = Word2Vec(min_count=3,
                  window=4,
                  vector_size=vector_size,
                  alpha=0.03, 
                  negative=10,
                  sample=0.001,
                  max_vocab_size=None,
                  max_final_vocab=vocab_size,
                  workers=8,
                  sg=1)
    
    vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                         sequence_length=maxlen,
                                         dtype='string')
    
    tokenized_corpus = vectorize_layer(corpus).numpy()
    tokenized_corpus = [[ by.decode('UTF-8') for by in s] for s in tokenized_corpus] # fixed #!!!
    tokenized_corpus = [" ".join(s) for s in tokenized_corpus]
    
    corpusPath = writeCorpus(tokenized_corpus)
    
    wv.build_vocab(corpus_file=corpusPath)
    wv.train(corpus_file=corpusPath,
             total_words=len(wv.wv),
             total_examples=wv.corpus_count, 
             epochs=15)
    vocab_list = list(wv.wv.key_to_index.keys())
    raw_embedding_matrix = wv.wv[vocab_list]
        
    for p in paths: os.remove(p)
    os.remove(temp_path + '/vocab.txt')
    os.remove(corpusPath)
        
    return wv, vocab_list, raw_embedding_matrix

def Word2VecVocabulary(corpus: pd.Series,
                       vector_size: int,
                       maxlen: int,
                       vocab_size: int = None) -> list[str]:
    # Word2Vec model:
    wv = Word2Vec(min_count=3,
                  window=4,
                  vector_size=vector_size,
                  alpha=0.03, 
                  negative=10,
                  sample=0.001,
                  max_vocab_size=None,
                  max_final_vocab=vocab_size,
                  workers=8,
                  sg=1)
    
    corpusPath = writeCorpus(corpus)
    
    # Build vocabulary and Train:
    wv.build_vocab(corpus_file=corpusPath)
    wv.train(corpus_file=corpusPath,
             total_words=len(wv.wv),
             total_examples=wv.corpus_count, 
             epochs=15) #!!!
    vocab_list = list(wv.wv.key_to_index.keys())
    raw_embedding_matrix = wv.wv[vocab_list]
    
    os.remove(corpusPath)
    
    return wv, vocab_list, raw_embedding_matrix

#%% Test

if __name__ == '__main__':
    df = pd.read_csv('./Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv',nrows=268346) #!!!
    
    vocab_size = 11_560
    vector_size = 300
    max_seq_len = 100
    
    # Test WordPieceVocabulary ================================================
    wv, vocab_list, raw_embedding_matrix = WordPieceVocabulary(df['text'], 
                                                               vector_size, 
                                                               max_seq_len, 
                                                               vocab_size)
    vocab_size = len(vocab_list)+2
    
    #Fullfill vocabulary:
    for token in ['[UNK]','[PAD]']: # include new tokens
        vocab_list.insert(0,token)
    # Add Special tokens to embedding matrix:
    PADvector = np.zeros((1,vector_size))
    UNKvector = np.zeros((1,vector_size))
    embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
    
    vectorize_layer = WordPieceTokenizer(vocabulary=vocab_list,
                                         sequence_length=max_seq_len)
    
    print("Tokenization layer...")
    t = time()
    trainData1 = vectorize_layer(df['text']).numpy()
    print(f"(Elapsed time: {strftime('%M:%S', gmtime(time()-t))})")
    # =========================================================================
    
    vocab_size = 30000
    vector_size = 300
    max_seq_len = 100
    
    # Test Word2VecVocabulary =================================================
    wv, vocab_list, raw_embedding_matrix = Word2VecVocabulary(df['text'], 
                                                              vector_size, 
                                                              max_seq_len, 
                                                              vocab_size)
    vocab_size = len(vocab_list)+2
    
    vectorize_layer = TextVectorization(max_tokens=vocab_size,
                                        standardize=None,
                                        output_mode='int',
                                        output_sequence_length=max_seq_len,
                                        vocabulary=vocab_list)
    
    #Fullfill vocabulary:
    for token in ['[UNK]','']: # include new tokens
        vocab_list.insert(0,token)
    # Add Special tokens to embedding matrix:
    PADvector = np.zeros((1,vector_size))
    UNKvector = np.zeros((1,vector_size))
    embedding_matrix = np.concatenate([PADvector, UNKvector, raw_embedding_matrix])
    
    print("Tokenization layer...")
    t = time()
    trainData2 = vectorize_layer(df['text']).numpy()
    print(f"(Elapsed time: {strftime('%M:%S', gmtime(time()-t))})")
    # =========================================================================

    # Length comparison
    count1 = np.sum((trainData1 > 0).astype('int32'),axis=1)
    count2 = np.sum((trainData2 > 0).astype('int32'),axis=1)
    
    diff = count1 - count2
    avg_diff = np.mean(diff)
    """
    WordPieceTokenizer always have more number of tokens per sentence than 
    TextVectorization:
    
    vocab_size=30000 and maxlen=100 --> 3.86 tokens in average
    vocab_size=11560 and maxlen=100 --> 5.47 tokens in average
    vocab_size=30000 and maxlen=128 --> 4.12 tokens in average
    vocab_size=11560 and maxlen=128 --> 5.91 tokens in average
    """

    








