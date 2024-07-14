import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def load_dataset(filename,sample=1.0):
    # Files ID from shared Drive:
    file_id = {
        'crossValidationTRAIN.csv':'1IZlEez0bcjACUStP0jmNiYl0bGQ87E2f',
        'crossValidationTEST.csv':'15AIhHxG2cbq6tn81nTugKM5s2rpEpmPZ',
        'english.csv':'1XpL4e8G49fG1QD8ui2izFekbyq6fKc6Y',   #
        'reviews_new.csv':'1MY_83a0ecvO3pV1ltobjoQIRGopSe64Z', #
        'PTReviewSample200K.csv':'1NUgXFj-jhoEsiaM8gdwMz64qopYejhgH', #
        'TEST_PT_3class_clean.csv':'1wzQ0Y92NG_ptO-17URAufU6pmQhpy8t0', #
        'TRAIN_PT_3class_clean.csv':'1zm5ZDXkXLL13OmSmABP-HvNu58iNWC-D', #
        'VALID_PT_3class_clean.csv':'1So6RLts87hA3Xyvk--pHjMfYpolmB5C3', #
        'PTReviewSample500K.csv':'1j0UmuVRuwn0FsaRpDSndiEQOi0ls9jhS',
        'PTbalanced.csv':'1TYYCa9CxC5grmfGmgFddOvL-tuCHnI7E' #
               }
    fid = file_id[filename]
    url='https://drive.google.com/uc?export=download&id=' + fid
    
    if sample == 1.0:
        return pd.read_csv(url)
    else:
        return pd.read_csv(url).sample(frac=sample,ignore_index=True)

def plot_ratings(df, min_rating=1, max_rating=5, n_labels=5):
    """
    Receives a dataframe with two columns: [text, rating]
    Plots a histogram of distribution of ratings
    """
    ratings = np.unique(df.rating)
    min_rating = ratings[0]
    max_rating = ratings[-1]
    _, ax = plt.subplots(figsize=(19.2,10.8), dpi=300)
    sns.countplot(data = df, x='rating', ax=ax)
    sns.set(font="Times New Roman")
    plt.title("Ratings distribution")
    plt.xlabel(f'Review Ratings ({min_rating} to {max_rating})')
    plt.ylabel("Number of occurrences")
    plt.show()
    
def show_train_history(train_history,metric,es=None):
    train_metric = metric
    validation_metric = "val_" + metric
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('Epoch')
    if es != None: 
        plt.xticks(np.linspace(1,es,es))
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.show()
    
def balance_df(df, frac=0.7, save_rest=False, min_label=1, max_label=5, n_labels=5, plot=True):
    """
    Receives a dataframe with two columns: [text, rating]
    Select random samples from df and trunk it to have a lenght equal to the
    less commom rating
    Plots results
    Returns a tuple: new balanced DataFrame, rest of DataFrame
    save_rest: if True, also returns the rest of the original DataFrame
    """
    if save_rest and frac==1.0:
        raise ValueError("Can't set frac=1.0 and save_test=True simultaneusly")
    if frac > 1.0:
        raise ValueError("frac can't be high than 1. It's a fraction")
    labels = np.linspace(min_label, max_label, n_labels)
    # Find the lowest class lenght:
    max_len = len(df)
    for i in labels:
        lenght = len(df[df.rating == i])
        if lenght < max_len: 
            max_len = lenght 
    max_len = int(frac*max_len) # to garantee rest of all classes
    
    # Create new balanced dataset:
    df1 = pd.DataFrame(columns=['text','rating'])
    non_taken = pd.DataFrame(columns=['text','rating'])
    for i in labels:
        new_rows = df[df.rating == i].sample(max_len)#.reset_index(drop=True)
        df1 = pd.concat([df1, new_rows], ignore_index=False)
        
        if save_rest:
            # Saves the rest of the dataset:
            taken = list(new_rows.index)
            taken.sort()
            aux = all_other_indexes(taken,len(df))
            try:
                df_aux = df.loc[aux]                  # df original sem as linhas sampleadas
                df_aux = df_aux[df_aux.rating == i]  # df sem linhas sampleadas apenas com rating i
                non_taken = pd.concat([non_taken, df_aux], ignore_index=True)
            except:
                print("Function balance_df() needs a `df` parameter with indexes of from:")
                print("RangeIndex(start=0, stop=len(df), step=1)")
                print("Invalid input lead to KeyError exception")

    percent = 100 - round((len(df)-len(df1))/len(df)*100,1)
    print(f"Using {round(percent,1)}% of dataset")
    if plot:
        # Calculate rating distribution:
        ratings = {'df':Counter(df.rating),'df1':Counter(df1.rating)}
        # Plot results
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(25, 10), sharey=True)
        fig.suptitle("Ratings distribution (Portuguese reviews)")
        fig.figsize=(16,20)
        # Original DataFrame:
        ax1.bar(ratings['df'].keys(), ratings['df'].values(), color='#0c7cbc', width=0.3)
        ax1.set(xlabel=f'Ratings ({min_label} to {max_label})', xticks=labels)
        # New DataFrame:
        ax2.bar(ratings['df1'].keys(), ratings['df1'].values(), color='#0c7cbc', width=0.3)
        ax2.set(xlabel=f'Ratings ({min_label} to {max_label})', xticks=labels)
        #fig.show()
    
    return df1.sort_index(), non_taken

def remove_tokens(sentence, remove_list, mode=True):
    """
    Receives a list of sentences tokenized.
    mode: if False -> remove if token is in remove_list
          if True  -> remove if token is not in remove_list 
    """
    return [token for token in sentence if (token in remove_list) == mode]

def calc_max_seq_len(features):
    """
    Receives a list of sentences tokenized.
    Calculates the maximum lenght of the tokenized sequences used for training
    RETURN: 
    """
    max_seq_len = mean_seq_len = 0
    index = max_index = 0
    for sample in features:
        mean_seq_len += len(sample)
        if len(sample)> max_seq_len:
            max_seq_len = len(sample)
            max_index = index
        index += 1
    mean_seq_len /= len(features)
    print(f"\nThe sequence of index {max_index} has a maximum lenght of {max_seq_len}")
    print(f"Avarage lenght of sentences is {round(mean_seq_len)}")
    return max_seq_len, max_index

def rem_seq(features,labels,max_len=800):
    """
    Receives a list of list of tokens. Lenght of lists of tokens are diverse
    Removes all list of list which lenght is greater than max_len
    Removes all list of list which lenght is 0
    Returns a pointer of features and the corresponding labels
    """
    labels = labels.tolist()
    # find indexes:
    for i in range(len(features))[::-1]:
        if len(features[i])>max_len or len(features[i])==0:
            del features[i]
            del labels[i]
    return features,np.asarray(labels)

def count_huge(features,cnt):
    """
    Receives a counter object
    Counts the number of sentences with more than 100 words
    Returns a Counter() object
    """
    for seq in features:
        if len(seq) > 100:
            cnt["> 100"] += 1
        if len(seq) > 200:
            cnt["> 200"] += 1
        if len(seq) > 300:
            cnt["> 300"] += 1
        if len(seq) > 400:
            cnt["> 400"] += 1
        if len(seq) > 500:
            cnt["> 500"] += 1
        if len(seq) > 600:
            cnt["> 600"] += 1
        if len(seq) > 700:
            cnt["> 700"] += 1
        if len(seq) > 800:
            cnt["> 800"] += 1
        if len(seq) > 900:
            cnt["> 900"] += 1
        if len(seq) > 1000:
            cnt["> 1000"] += 1 
    return cnt

def print_wordcloud(seq_list, max_tokens=10000):
    """
    seq_list: list of list of tokens
    max_tokens: Maximum number of tokens to use for WordCloud
    Prints a WordCloud of max_tokens taken from seq_list randomly
    """
    def get_text_cloud(seq_list):
        return " ".join(" ".join(word for word in seq) for seq in seq_list)
    
    text_cloud = get_text_cloud(random.sample(seq_list, max_tokens))
    word_cloud = WordCloud(max_font_size = 250, width = 1520, height = 1000)
    # print:
    word_cloud.generate(text_cloud)
    plt.figure(figsize = (16, 9))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()
    
def all_other_indexes(lst, maxlen): # optimized
    """ 
    Receives a list of ordered integer values, from lower to greater
    Receives the maximum lenght of the output: maxlen - len(lst)
    Returns a list of all integer values missing within lst, from 
    0 to maxlen
    """
    new_lst = []
    finish = False
    for index in range(maxlen):
        if finish:
            new_lst.append(index)
            continue 
        try: # Takes first value of 'taken'
            next_i = lst[0] 
        except:
            finish = True
            new_lst.append(index)
            continue
        
        if index != next_i:
            new_lst.append(index)
        else:
            lst = lst[1:]
    return new_lst

def merge_half_rating(df,mode='floor'):
    """
    Receives a dataframe with two columns: [text, rating]
    Merges half ratings with next or previous (floor or ceil)
    """
    merge_func = eval(f"np.{mode}")
    df['rating'] = merge_func(df['rating']).astype('int32')

    plot_ratings(df)


def merge_pos_neg(df,threshold=3.5):
    """
    Receives a dataframe with two columns: [text, rating]
    Join ratings separated by threshold
    """
    polarity = (df['rating'] > threshold).astype('int32')
    df['rating'] = polarity
    plot_ratings(df)
    
def merge_pos_neg_neutral(df, plot=True):
    """
    Receives a dataframe with two columns: [text, rating]
    Join ratings separated by threshold
    class0 : Negative
    class1 : Neutral
    class2 : Positive
    """
    df3 = df.copy()
    ratings = df3['rating']
    
    class0 = df3[ratings<=2.0]['rating'].apply(lambda x: 0)
    class1 = df3[np.logical_and(ratings>=2.5, ratings<=3.5)]['rating'].apply(lambda x: 1)
    class2 = df3[ratings>=4.0]['rating'].apply(lambda x: 2)
    
    df3['rating'] = pd.concat([class0,class1,class2]).sort_index()
    if plot: plot_ratings(df3)
    return df3
    
def store_results(global_values, particular_values, Results_path, Model_ID):
    " Deprecated "
    per_class_values = {}
    for name, metrics in particular_values.items():
        for i,metric in enumerate(metrics):
            per_class_values[f"{name}_{i}"] = metric
            
    global_values.update(per_class_values)
    results = pd.DataFrame(global_values,index=[Model_ID])
    
    # Check if file already exist
    files_in_path = list(Path(Results_path).glob('*.csv'))
    files_in_path = [file._cparts[-1] for file in files_in_path]
    # if already exists, concatenate results
    if "models_comparison.csv" in files_in_path:
        prev_results = pd.read_csv(f"{Results_path}models_comparison.csv",index_col=0)
        results = pd.concat([prev_results, results])
    
    # load new results to file
    results.to_csv(f"{Results_path}models_comparison.csv",mode='w')
    

def join_negative(text):
    """ (not used)
    Example of usage:
    df['text'] = df['text'].apply(join_negative)
    """
    text = text.split()
    jump = 0
    joined_text = []
    
    for i in range(len(text)):
        if jump: 
            jump = 0
            continue
        bigram = text[i:i+2]
        
        if bigram[0] in ['não','nao','sem','muito','muita','muitos','muitas']:
            joined_text.append('_'.join(bigram))
            jump = 1
        else:
            joined_text.append(text[i])
            jump = 0
  
    return ' '.join(joined_text)




