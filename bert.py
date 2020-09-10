import requests
import json
import pandas as pd
import numpy as np
# import csv
from flashtext import KeywordProcessor
from scipy.spatial.distance import cdist
# from langdetect import detect
import re
# import sys
# from nltk.stem.porter import PorterStemmer
use_stemmer = False
data_vector = []
global url
global headers
url = 'http://192.168.1.26:8125/encode'
headers = {'content-type': 'application/json'}

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = porter_stemmer.stem(word)
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def preprocess_tweets(tweetlist):
    tweets = []
    for n in range(len(tweetlist)):
        line = str(tweetlist[n])
        tweet_id = line[:line.find(',')]
        line = line[1 + line.find(','):]
        tweet = line
        processed_tweet = preprocess_tweet(tweet)
        tweets.append('%s' % (processed_tweet))
    return tweets

#This function requests the ML server to vectorize all the data and return the data vector
def search_pp(arr): 
    data_vector = []
    a = []
    for x in arr:
        try:
            a.append(x)
            data = {"texts": a, "id": 123, "is_tokenized": False}
            r = requests.post(url, headers=headers, json=data)
            r = r.json()
            r = json.dumps(r)
            loaded_r = json.loads(r)
            result = loaded_r["result"]
            data_vector.append(result)
        except:
            pass
        del a[0:]
    return data_vector

# def query_pp(q):
#     vec = []
#     query = str(q)
#     d_array = []
#     d_array.append(query)
#     query_data = {"texts": d_array, "id": 124, "is_tokenized": False}
#     r = requests.post(url, headers=headers, json=query_data)
#     r = r.json()
#     r = json.dumps(r)
#     new_loaded_r = json.loads(r)
#     new_result = new_loaded_r["result"]
#     # query_vec = vec_generator(query)
#     return new_result


#calculates manhattan distance and returns the scores between the query and each data array
def get_score(q, d):  
    q = np.array(q)
    d = np.array(d)
    topk = 3
    score_array = []
    d_arr = []
    for i in d:
    # compute normalized dot product as score
        # score = np.sum(q * i, axis=1) / np.linalg.norm(i, axis=1)
        score = cdist(q, i, metric='cityblock')
        topk_idx = np.argsort(score)[::-1][:topk]
        d_arr.append(i)
        for idx in topk_idx:
            score_array.append(score[idx])
    
    return score_array

#returns the data according to the index provided
def get_result(arr, m_index):
    for idx, item in enumerate(arr):
        if idx == m_index[0]:
            return item

#returns processed query
def handle_my_query(query_tittle, query_brand=None, query_description=None):
    q = []
    if query_description is None and query_brand is not None:
        q.append(str(query_tittle))
        q.append(str(query_brand))
        
    elif query_description is None and query_brand is None:
        q.append(str(query_tittle))
        
    elif query_description is not None and query_brand is None:
        q.append(str(query_tittle))
        q.append(str(query_description))
    else:
        query_desc = []
        q.append(str(query_tittle))
        q.append(str(query_brand))
        query_desc.append(query_description)
        processed_query_description = preprocess_tweets(query_desc)
        q.append(processed_query_description[0])
        
    return q


def get_all_data(df, query):        #concatenates everything data array+query array together
    df_tittle = []
    df_brand = []
    df_description = []
    for i in df:
        for idx,j in enumerate(i):
            if idx==0:
                df_tittle.append(j)
            elif idx==1:
                df_brand.append(j)
            elif idx==2:
                df_description.append(j)
    
    df_tittle = np.array(df_tittle)
    df_brand = np.array(df_brand)
    df_description = np.array(df_description)
    t_arr = df_tittle.astype('str')
    b_arr = df_brand.astype('str')
    d_arr = df_description.astype('str')
    processed_d_arr = preprocess_tweets(d_arr)
    total_arr = []
    total_arr.append(np.array(t_arr))
    total_arr.append(np.array(b_arr))
    total_arr.append(np.array(processed_d_arr))
    total_arr.append(query)
    return total_arr


def vectorize_them(total_arr):              #concatenates the total array on axis 0 and calls search pp(Line no 95) for vectorizing everything
    total_arr = np.concatenate(total_arr, axis=0)
    total_arr = np.array(total_arr)
    total_vec = search_pp(total_arr)
    return total_vec


def get_the_result(total_vec, length_of_q, limit, b_query_i=None, d_query_i=None):  #returns the index of the best match
    if b_query_i is False and d_query_i is False:
        length_of_q = -(length_of_q)
        q_vec = total_vec[length_of_q:]
        t_q_vec = q_vec[0]
        t_arr_vec = []
        for idx, item in enumerate(total_vec):
            if idx > len(total_vec)+length_of_q-1:
                break
            if idx < limit:
                t_arr_vec.append(item)
                
        t_score = get_score(t_q_vec, t_arr_vec)
        total_scores = np.array(t_score)
        max_total = np.amin(t_score)
        # print("Score  :" , max_total)
        max_index = np.where(t_score == np.amin(t_score))
        return max_index
    
    elif b_query_i is False and d_query_i is True:
        length_of_q = -(length_of_q)
        q_vec = total_vec[length_of_q:]
        t_q_vec = q_vec[0]
        d_q_vec = q_vec[1]
        t_arr_vec = []
        d_arr_vec = []
        for idx, item in enumerate(total_vec):
            if idx > len(total_vec)+length_of_q-1:
                break
            if idx < limit:
                t_arr_vec.append(item)
            elif idx>=limit and idx< limit+limit:
                d_arr_vec.append(item)
                
        t_score = get_score(t_q_vec, t_arr_vec)
        d_score = get_score(d_q_vec, d_arr_vec)
        total_scores = []
        for t, d in zip(t_score, d_score):
            total_scores.append((t+d)/2)
        total_scores = np.array(total_scores)
        max_total = np.amin(total_scores)
        # print("Average Score  :", max_total)
        max_index = np.where(total_scores == np.amin(total_scores))
        return max_index
        
    elif b_query_i is True and d_query_i is False:
        length_of_q = -(length_of_q)
        q_vec = total_vec[length_of_q:]
        t_q_vec = q_vec[0]
        b_q_vec = q_vec[1]
        t_arr_vec = []
        b_arr_vec = []
        for idx, item in enumerate(total_vec):
            if idx > len(total_vec)+length_of_q-1:
                break
            if idx < limit:
                t_arr_vec.append(item)
            elif idx>=limit and idx< limit+limit:
                b_arr_vec.append(item)
        b_score = get_score(b_q_vec, b_arr_vec)
        t_score = get_score(t_q_vec, t_arr_vec)
        total_scores = []
        for b, t in zip(b_score, t_score):
            total_scores.append((b+t)/2)
        total_scores = np.array(total_scores)
        max_total = np.amin(total_scores)
        # print("Average Score  :" , max_total)
        max_index = np.where(total_scores == np.amin(total_scores))
        return max_index
    
    else:   
        length_of_q = -(length_of_q)
        q_vec = total_vec[length_of_q:]
        t_q_vec = q_vec[0]
        b_q_vec = q_vec[1]
        d_q_vec = q_vec[2]
        b_arr_vec = []
        t_arr_vec = []
        d_arr_vec = []
        for idx, item in enumerate(total_vec):
            if idx > len(total_vec)+length_of_q-1:
                break
            if idx < limit:
                t_arr_vec.append(item)
            elif idx > limit-1 and idx < limit+limit:
                b_arr_vec.append(item)
            else:
                d_arr_vec.append(item)
                
        b_score = get_score(b_q_vec, b_arr_vec)
        t_score = get_score(t_q_vec, t_arr_vec)
        d_score = get_score(d_q_vec, d_arr_vec)
        total_scores = []
        for b, t, d in zip(b_score, t_score, d_score):
            total_scores.append((b+t+d)/3)

        total_scores = np.array(total_scores)
        max_total = np.amin(total_scores)
        # print("Average Score  :", max_total)
        max_index = np.where(total_scores == np.amin(total_scores))
        return max_index
    


def final_result(df, result_index, b_query_i, d_query_i):          #returns the corresponding data based in the index of best match
    df_tittle = []
    df_brand = []
    df_description = []
    for i in df:
        for idx, j in enumerate(i):
            if idx == 0:
                df_tittle.append(j)
            elif idx == 1:
                df_brand.append(j)
            elif idx == 2:
                df_description.append(j)
    df_tittle = np.array(df_tittle)
    df_brand = np.array(df_brand)
    df_description = np.array(df_description)
               
    if b_query_i is False and d_query_i is False:
        t_arr = df_tittle.astype('str')
        string = []
        string.append(get_result(t_arr, result_index))
        return string
    
    elif b_query_i is False and d_query_i is True:
        t_arr = df_tittle.astype('str')
        d_arr = df_description.astype('str')
        string = []
        string.append(get_result(t_arr, result_index))
        string.append(get_result(d_arr, result_index))
        return string
        
    elif b_query_i is True and d_query_i is False:
        t_arr = df_tittle.astype('str')
        b_arr = df_brand.astype('str')
        string = []      
        string.append(get_result(t_arr, result_index))
        string.append(get_result(b_arr, result_index))
        return string
    else:
        t_arr = df_tittle.astype('str')
        b_arr = df_brand.astype('str')
        d_arr = df_description.astype('str')
        string = []
        string.append(get_result(t_arr, result_index))
        string.append(get_result(b_arr, result_index))
        string.append(get_result(d_arr, result_index))
        return string

def crosscheck(sub_category, result):               #checks if the subcategory string exists, returns the data if yes or False if not
    keyword_processor = KeywordProcessor()
    for i in sub_category:
        keyword_processor.add_keyword(i)
    
    concetanate_all = ""
    for i in result:
        concetanate_all = concetanate_all+" "+str(i)
    
    keywords_found = keyword_processor.extract_keywords(concetanate_all)
    true_count = 0
    for item in sub_category:
        if item in keywords_found:
            true_count = true_count+1
    if true_count>= len(sub_category):
        return True
    else:
        return False


def bert_test_main(query, df, sub_category):             #its the main function, you will call this one
    query_tittle = query[0]
    try:
        query_brand = query[1]
    except IndexError:
        query_brand = ""
    try:
        query_description = query[2]
    except IndexError:
        query_description = ""
    b_query_i = True
    d_query_i = True

    if len(query_description) < 5 and len(query_brand) >= 1:
        query = handle_my_query(query_tittle, query_brand)
        d_query_i = False
    elif len(query_description) > 5 and len(query_brand) < 1:
        query = handle_my_query(query_tittle, query_description)
        b_query_i = False
    elif len(query_description) < 5 and len(query_brand) < 1:
        query = handle_my_query(query_tittle)
        d_query_i = False
        b_query_i = False
    else:
        query = handle_my_query(query_tittle, query_brand, query_description)

    # combining all the data array and query array
    total_arr = get_all_data(df, query)
    # running the ML for vectorizing everything
    total_vec = vectorize_them(total_arr)
    # its for finding the length of each column
    limit = (len(total_vec)-len(query))/3
    limit = int(limit)
    result_index = get_the_result(total_vec, len(query), limit, b_query_i, d_query_i)  # finds the index of the best match
    result_index = np.asarray(result_index)
    # finds the tuple according to the index
    result = final_result(df, result_index, b_query_i, d_query_i)
    # its for validating the results if the results contain our expected category
    f_result = crosscheck(sub_category, result)
    if f_result is True:
        return result_index[0]
    else:
        return False
