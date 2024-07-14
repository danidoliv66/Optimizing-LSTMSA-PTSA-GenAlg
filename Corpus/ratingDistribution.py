# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:44:12 2024

@author: danie
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# colors = sns.color_palette('pastel')[0:9]

# Load dataset
df = pd.read_csv('./Data/ZomatoFinal/FULL_PT_100_ImBalanced.csv',
                  usecols=['raw_text', 'rating','review_id'])

#%% Functions
def plot_ratings(df):
    """
    Plots a histogram of distribution of ratings
    """
    def rem(x):
        frac, _ = math.modf(x)
        return "Yes" if bool(frac) else "No"
    
    def to_3class(df):
        ratings = df['rating']
        class0 = df[ratings<=2.0]['rating'].apply(lambda x: 0)
        class1 = df[np.logical_and(ratings>=2.5, ratings<=3.5)]['rating'].apply(lambda x: 1)
        class2 = df[ratings>=4.0]['rating'].apply(lambda x: 2)
        return pd.concat([class0,class1,class2]).sort_index()
    
    _, (ax1, ax2) = plt.subplots(1,2, figsize=(17,8), dpi=120)
    # ax1.get_shared_y_axes().join(ax2, ax1)
    
    # Merged ratings (sentiments) =============================================
    df['sentiment'] = to_3class(df)
    sentiment = np.unique(df.sentiment)
    min_rating = sentiment[0]
    max_rating = sentiment[-1]
    
    sns.countplot(data = df, x='sentiment',hue=None, color='#4c72b0', ax=ax2)
    sns.set(font="Times New Roman")
    ax2.set_xticklabels(['Negative','Mixed','Positive'], fontsize=20)
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax2.get_yticks()]
    ax2.set_yticklabels(formatted_yticks, fontsize=19)
    ax2.set_xlabel('Sentiment', fontsize=20)
    ax2.set_ylabel("", fontsize=20)
    ax2.grid(True,axis='y',linestyle='dotted',color='k')
    
    # Original ratings ========================================================
    ratings = np.unique(df.rating)
    min_rating = ratings[0]
    max_rating = ratings[-1]
    df['half'] = df.rating.apply(rem)

    
    sns.countplot(data = df, x='rating',hue='half', dodge=False, ax=ax1)
    sns.set(font="Times New Roman")
    formatted_xticks = [(tick+2)/2 for tick in ax1.get_xticks()]
    ax1.set_xticklabels(formatted_xticks, fontsize=19)
    formatted_yticks = [f"{tick//1000:.0f}k" if tick!=0 else "0" for tick in ax1.get_yticks()]
    ax1.set_yticklabels(formatted_yticks, fontsize=19)
    ax1.set_xlabel(f'Star ratings', fontsize=20)
    ax1.set_ylabel("Number of occurrences", fontsize=20)
    ax1.grid(True,axis='y',linestyle='dotted',color='k')
    ax1.legend(title='Half rating?' ,loc='upper left', 
               fontsize=14, title_fontsize=15)
    
    plt.show()
    print("Percentages (ratings):",
          "1.0*:  {:.2f}%".format( ((df.rating == 1.0).sum()/len(df))*100 ),
          "1.5*:  {:.2f}%".format( ((df.rating == 1.5).sum()/len(df))*100 ),
          "2.0*:  {:.2f}%".format( ((df.rating == 2.0).sum()/len(df))*100 ),
          "2.5*:  {:.2f}%".format( ((df.rating == 2.5).sum()/len(df))*100 ),
          "3.0*: {:.2f}%".format( ((df.rating == 3.0).sum()/len(df))*100 ),
          "3.5*: {:.2f}%".format( ((df.rating == 3.5).sum()/len(df))*100 ),
          "4.0*: {:.2f}%".format( ((df.rating == 4.0).sum()/len(df))*100 ),
          "4.5*: {:.2f}%".format( ((df.rating == 4.5).sum()/len(df))*100 ),
          "5.0*: {:.2f}%".format( ((df.rating == 5.0).sum()/len(df))*100 ),
          sep='\n')
    
    print("\nPercentages (sentiments):",
          "Negative:  {:.2f}%".format( ((df.sentiment == 0).sum()/len(df))*100 ),
          "Neutral:  {:.2f}%".format( ((df.sentiment == 1).sum()/len(df))*100 ),
          "Positive: {:.2f}%".format( ((df.sentiment == 2).sum()/len(df))*100 ),
          sep='\n')
    
    
#%% Plot ratings

plot_ratings(df)