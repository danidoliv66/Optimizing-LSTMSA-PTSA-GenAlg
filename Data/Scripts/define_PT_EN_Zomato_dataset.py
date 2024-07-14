# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:07:43 2024

@author: Admin
"""

import pandas as pd
import numpy as np

df = pd.read_csv("./Data/Batch5Final.csv")
print(*df.columns,sep='\n')

# Remove Marcos's columns
to_drop = ["name_gender", "first_name", "word_count",
           "text_no_stopword", "text_stemm", 
           "pt_senti_positive", "pt_senti_negative", 
           "en_senti_positive", "en_senti_negative", 
           "emotaix_positive", "emotaix_neutral", 
           "emotaix_negative", "emotaix_unknown"]

df_clean = df.drop(to_drop, axis=1)

print(*df_clean.columns, sep='\n')

# Number of reviews
N = len(df_clean)
N_empty = df_clean.text.isna().sum()
N_with_text = N - N_empty
print("Number of samples:", N)
print("Number of empty reviews:", N_empty)

# Language distribution

no_EN = (df_clean.text_language == 'en').sum()
no_PT = (df_clean.text_language == 'pt').sum()
no_Other = N_with_text - no_EN - no_PT

print("Number of english (EN) reviews:", no_EN, f"({100*no_EN/N_with_text:.1f}%)")
print("Number of portuguese (PT) reviews:", no_PT, f"({100*no_PT/N_with_text:.1f}%)")
print("Other languages:", no_Other, f"({100*no_Other/N_with_text:.1f}%)")

if False:
    df_clean.to_csv("./Data/PT-EN Zomato Dataset.csv", index=False)





