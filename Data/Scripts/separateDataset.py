# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:54:10 2023

@author: danie
"""
#%% Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools import plot_ratings

#%% Load Dataset to apply separation
""" 
Datasets to apply separation
    "./Data/FilteredBatch5Final.csv" ## Original Zomato dataset
    "./Data/TEMP/UND_temp.csv" ## Dataset after Under sampling

"""
file = "./Data/FilteredBatch5Final.csv"
# file = "./Data/TEMP/UND_temp.csv"

df = pd.read_csv(file)
seed = 2345728788

df_train, df_test = train_test_split(df, test_size=0.30, 
                                     random_state=seed, 
                                     shuffle=True,
                                     stratify=df['rating'])

df_train, df_val = train_test_split(df_train, test_size=0.15,
                                    random_state=seed,
                                    shuffle=True,
                                    stratify=df_train['rating'])

print(f"Training set: {len(df_train)/len(df)*100:.2f}% or {len(df_train)} samples")
print(f"Validation set: {len(df_val)/len(df)*100:.2f}% or {len(df_val)} samples")
print(f"Testing set: {len(df_test)/len(df)*100:.2f}% or {len(df_test)} samples")

#%% Fix index
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

#%% Plot results
plot_ratings(df)
plot_ratings(df_train)
plot_ratings(df_test)
plot_ratings(df_val)

#%% Save splits in files (Change depending the case)
if False:
    # For "./Data/FilteredBatch5Final.csv":
    df.to_csv("./Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv",index=False)
    df_train.to_csv("./Data/ZomatoFinal/TRAIN_PT_60_ImBalanced.csv",index=False)
    df_test.to_csv("./Data/ZomatoFinal/TEST_PT_30_ImBalanced.csv",index=False)
    df_val.to_csv("./Data/ZomatoFinal/VALID_PT_10_ImBalanced.csv",index=False)
    
    # # For "./Data/TEMP/UND_temp.csv":
    # df.to_csv("./Data/ZomatoFinal/FULL_PT_100_UNDBalanced.csv",index=False)
    # df_train.to_csv("./Data/ZomatoFinal/TRAIN_PT_56_UNDBalanced.csv",index=False)
    # df_test.to_csv("./Data/ZomatoFinal/TEST_PT_30_UNDBalanced.csv",index=False)
    # df_val.to_csv("./Data/ZomatoFinal/VALID_PT_14_UNDBalanced.csv",index=False)

