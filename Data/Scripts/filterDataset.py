# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:44:30 2023

@author: danie
"""
#%% Import libraries
import pandas as pd
import numpy as np
from Cleaning.textPreprocessing import clean

#%% Load dataset and select columns
df = pd.read_csv("./Data/Batch5Final.csv")
# print(*df.columns,sep='\n')

use_columns = ['review_id','res_id',
               'text_translation_pt',
               'rating','res_rating',
               'text_language'
               ]

filter_df = (df[use_columns].dropna(subset=['text_translation_pt'])
                           .drop_duplicates(subset=['text_translation_pt']))

print("NaN count:", filter_df.isna().sum().text_translation_pt)
filter_df.reset_index(drop=True, inplace=True)
print(filter_df.index)

#%% Apply ALL cleaning steps (for quick usage):
steps = np.ones(7)
text_column = filter_df.text_translation_pt.apply(lambda x: clean(x, *steps))

# Remove empty reviews
text_column = text_column.replace('', np.nan)
print("NaN count:", text_column.isna().sum())

#%% Concatenate cleaned text to Dataset
filter_df = pd.concat([filter_df,pd.DataFrame({'text':text_column})], axis='columns')
filter_df = filter_df.rename(columns={'text_translation_pt':'raw_text'})
print("NaN count:", filter_df.isna().sum(), sep='\n')
filter_df = filter_df.dropna(subset=['text']).drop_duplicates(subset=['text'])
print("NaN count:", filter_df.isna().sum(), sep='\n')

#%% Save filtered Dataset in csv file
filter_df.to_csv("./Data/FilteredBatch5Final.csv", index=False)
# df_loaded = pd.read_csv("./Data/FilteredBatch5Final.csv")
