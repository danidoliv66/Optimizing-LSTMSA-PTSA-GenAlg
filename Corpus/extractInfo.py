# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:29:00 2024

@author: Daniel
"""
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Cleaning.textPreprocessing import STOPWORDS_PT

#%% Functions

def plot_words_dist(vocab, top=45):
    df = pd.DataFrame({"Words":vocab.keys(),
                       "Count":vocab.values()})
    df = df.sort_values(by='Count', ascending=False)[:top]
    
    # Original
    fig, ax = plt.subplots(figsize=(9.2,9.2), dpi=300)
    sns.barplot(data=df,x='Words',y='Count', 
                palette="crest",orient='v',
                ax=ax)
    # sns.set(font="Times New Roman")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                       rotation=42, ha='right'
                       )
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax.get_yticks()]
    ax.set_yticklabels(formatted_yticks, fontsize=19)
    ax.set_xlabel('', fontsize=1)
    ax.set_ylabel("Number of occurrences", fontsize=20)
    plt.grid(True, axis='y', linestyle='dotted',color='k')
    
    plt.show()

def plot_words_dist3(vocab1, vocab2, vocab3, top=45):
    df1 = pd.DataFrame({"Words":vocab1.keys(),
                        "Count":vocab1.values()})
    df1 = df1.sort_values(by='Count', ascending=False)[:top]
    
    df2 = pd.DataFrame({"Words":vocab2.keys(),
                        "Count":vocab2.values()})
    df2 = df2.sort_values(by='Count', ascending=False)[:top]
    
    df3 = pd.DataFrame({"Words":vocab3.keys(),
                        "Count":vocab3.values()})
    df3 = df3.sort_values(by='Count', ascending=False)[:top]
    
    # Original
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False,
                                        figsize=(30,9.2), dpi=300)
    # VOCAB 1
    sns.barplot(data=df1, x='Words',y='Count', 
                palette="crest", orient='v',
                ax=ax1)
    # sns.set(font="Times New Roman")
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=18,
                        rotation=42, ha='right')
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax1.get_yticks()]
    ax1.set_yticklabels(formatted_yticks, fontsize=19)
    ax1.set_xlabel('', fontsize=1)
    ax1.set_ylabel("Number of occurrences", fontsize=20)
    ax1.grid(True, axis='y', linestyle='--',color='k', linewidth=2)
    
    # VOCAB 2
    sns.barplot(data=df2, x='Words',y='Count', 
                palette="crest", orient='v',
                ax=ax2)
    # sns.set(font="Times New Roman")
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=18,
                        rotation=42, ha='right')
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax2.get_yticks()]
    ax2.set_yticklabels(formatted_yticks, fontsize=19)
    ax2.set_xlabel('', fontsize=1)
    ax2.set_ylabel("", fontsize=1)
    ax2.grid(True, axis='y', linestyle='--',color='k', linewidth=2)
    
    # VOCAB 1
    sns.barplot(data=df3, x='Words',y='Count', 
                palette="crest", orient='v',
                ax=ax3)
    # sns.set(font="Times New Roman")
    ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=18,
                        rotation=42, ha='right')
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax3.get_yticks()]
    ax3.set_yticklabels(formatted_yticks, fontsize=19)
    ax3.set_xlabel('', fontsize=1)
    ax3.set_ylabel("", fontsize=1)
    ax3.grid(True, axis='y', linestyle='--',color='k', linewidth=2)
    
    plt.show()

#%% From stats
stats = pd.read_csv("Corpus/stats.csv")

print("General stats:", stats[['token_count','punctuation_count','stopword_count']].describe(), sep='\n')
print()
print("Mean tokens: {:.0f}".format(stats.token_count.mean()) )
print("Median tokens: {:.0f}".format(stats.token_count.median()) )
print("> Puntuation: {:.1f}%".format((stats.punctuation_count/stats.token_count).mean()*100) )
print("> Stopword_count: {:.1f}%".format((stats.stopword_count/stats.token_count).mean()*100) )
print("Mean lemmatized tokens: {:.0f}".format(stats.lemma_count.mean()) )
print("Median lemmatized tokens: {:.0f}".format(stats.lemma_count.median()) )

#%% From vocabulary (original tokens)
with open("Corpus/vocab.pkl",'rb') as f:
    vocab = pickle.load(f)

total = len(vocab)
moreThan1 = len([v for k,v in vocab.items() if v > 1])
moreThan5 = len([v for k,v in vocab.items() if v > 5])
    
print(f"Total: {total} (100%)")
print(f"More than 1 in corpus: {moreThan1} ({100*moreThan1/total:.1f}%)")
print(f"More than 5 in corpus: {moreThan5} ({100*moreThan5/total:.1f}%)")

#%% From vocabulary (lemmatized tokens)
with open("Corpus/lemmas.pkl",'rb') as f:
    lemmas = pickle.load(f)

total = len(lemmas)
moreThan1 = len([v for k,v in lemmas.items() if v > 1])
moreThan5 = len([v for k,v in lemmas.items() if v > 5])
    
print(f"Total: {total} (100 %)")
print(f"More than 1 in corpus: {moreThan1} ({100*moreThan1/total:.1f} %)")
print(f"More than 5 in corpus: {moreThan5} ({100*moreThan5/total:.1f} %)")

#%% Plot word distribution (From vocab)
N = 21
# plot_words_dist(vocab, N)

vocab_noStopwords = {tok:count for tok,count in vocab.items() if tok.lower() not in STOPWORDS_PT}
# plot_words_dist(vocab_noStopwords, N)

vocab_noStopwords_noPunc = {tok:count for tok,count in vocab_noStopwords.items() if tok.isalpha()}
# plot_words_dist(vocab_noStopwords_noPunc, N)

plot_words_dist3(vocab,vocab_noStopwords,vocab_noStopwords_noPunc, N)

#%% Plot word distribution (From lemmas)
N = 21
# plot_words_dist(lemmas, N)

lemmas_noStopwords = {tok:count for tok,count in lemmas.items() if tok.lower not in STOPWORDS_PT}
# plot_words_dist(lemmas_noStopwords, N)

lemmas_noStopwords_noPunc = {tok:count for tok,count in lemmas_noStopwords.items() if tok.isalpha()}
# plot_words_dist(lemmas_noStopwords_noPunc, N)

plot_words_dist3(lemmas,lemmas_noStopwords,lemmas_noStopwords_noPunc, N)





