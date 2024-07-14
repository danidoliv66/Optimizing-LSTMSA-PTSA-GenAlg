# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:10:22 2024

@author: Daniel
"""
import json
import spacy
import pickle
import pandas as pd
from collections import Counter
from Cleaning.textPreprocessing import clean, STOPWORDS_PT

def is_not_blank(text):
    if text in ['', ' ']:
        return False
    # If it have more than a single space
    sub = text.split()
    if len(sub) == 0:
        return False
    # Not blank
    return True

def json_to_tokens():
    """parse json file to list of list (not padded)"""
    with open("Corpus/tokenized_data.json", 'r', encoding='utf-8') as f:
        tokenized_data = json.load(f)
        
    review_ids = list(map(int,tokenized_data.keys()))
    tokens = list(tokenized_data.values())
        
    return review_ids, tokens

# Load an NLP model trained on 'pt_core_news_lg' corpus
nlp = spacy.load("pt_core_news_lg")

# Load dataset
df = pd.read_csv('./Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv',
                  usecols=['raw_text', 'rating','review_id'])

#%% Extract stats
N = len(df) # Total
# N = 200     # Testing

# Clean text to remove formating errors
cleaner = lambda x: clean(x, *[0,0,0,0,0,0,0])
texts = df.raw_text.copy()[:N].apply(cleaner)

# Define accumulators
token_count = len(texts)*[0]
tokenized_texts = len(texts)*[[]]
lemma_count = len(texts)*[0]

stopword_count = len(texts)*[0]
is_stoword_texts = len(texts)*[[]]
punctuation_count = len(texts)*[0]

vocab = Counter()
lemmas = Counter()

for doc_id, document in enumerate(nlp.pipe(texts)):
    
    tokenized_doc = list(filter(is_not_blank, [tok.text.replace('\xa0', '') for tok in document]))
    tmp = list(filter(is_not_blank, [tok.lemma_.replace('\xa0', '') for tok in document]))
    lemmatized_doc = []
    for el in [tok.split() for tok in tmp]:
        lemmatized_doc.extend(el)
    
    vocab.update(tokenized_doc) # for EDA
    lemmas.update(lemmatized_doc) # to use in the models
    tokenized_texts[doc_id] = lemmatized_doc # NEED TO STORE LEMMETIZED TOKENS!!!
    
    token_count[doc_id] = len(tokenized_doc)
    lemma_count[doc_id] = len(lemmatized_doc)
    
    # This info is when considering original tokens, NOT LEMMAS
    is_stoword_texts[doc_id] = [tok.lower() in STOPWORDS_PT for tok in tokenized_doc] # for EDA
    stopword_count[doc_id] = sum(is_stoword_texts[doc_id])
    punctuation_count[doc_id] = sum([tok.pos_ == 'PUNCT' for tok in document])

#%% Save stats and tokens

## Save stats
data = dict(review_id=df.review_id.copy()[:N],
            token_count=token_count,
            stopword_count=stopword_count,
            punctuation_count=punctuation_count,
            lemma_count=lemma_count,)

stats = pd.DataFrame(data)
stats.to_csv("Corpus/stats.csv", index=False)
    
## Save tokenized data
tokenized_data = {k:v for k,v in zip(df.review_id.copy()[:N], tokenized_texts)}
with open("Corpus/tokenized_data.json", 'w', encoding='utf-8') as f:
    json.dump(tokenized_data, f, indent=1, ensure_ascii=False)

## Save Vocabulary size
with open("Corpus/vocab_size.txt",'w') as f:
    f.write(f"Total: {len(vocab)}\n")
    f.write(f"Lemmatized: {len(lemmas)}\n")

## Save Vocabulary
with open("Corpus/vocab.pkl",'wb') as f:
    pickle.dump(vocab, f)
with open("Corpus/lemmas.pkl",'wb') as f:
    pickle.dump(lemmas, f)
    
    