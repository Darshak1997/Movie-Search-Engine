# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:32:02 2019

@author: Darshak
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
#from num2words import num2words
import math
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
import operator


#N_List = []
#    for i in range(len(data_test)):
#        score_class = {}
#        prob.append(i)
#        score_class['Keys'] = i
#    for check in range(124):
#        N_List.append(1)
#        main_N = []
#        for n in range(100):
#            main_N.append(n)

def get_results(k1, query):
#    print("I am getting you reults")
    global Score_TF_IDF
    initialize()
    if os.path.isfile("image_tf_idf.pkl"):
        Score_TF_IDF = pickle.load(open('image_tf_idf.pkl', 'rb'))
    else:
#        print("In else of get_scores:")
        build()
    return Final_Score(k1, query)


def initialize():
    
    global movie_col, noise_list, movie_row, N, tokenizer, stopword, stemmer, inverted_index, document_vector, lemmatizer, stemmer_snow, DF1, Score_TF_IDF

    movie_row = pd.read_csv('Caption_Images.csv')
    
    N = 125
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    stemmer_snow = SnowballStemmer("english")
#    lemmatizer = WordNetLemmatizer()

    inverted_index = {}
    document_vector = {}
    
    print("Initialized")
    
#tf_idf1
def build():
    global Score_TF_IDF
    print("I'm Build")
    N_List = []
    prob = []
    Processed_Word = []
    # processed_title=[]
    total = 104
    for i in range(len(movie_row)):
        new_string = str(data_pre_processing(movie_row['caption'][i]))
        Processed_Word.append(word_tokenize(new_string))
    print(Processed_Word[0])
    DF1 = {}
    score_tf_idf = len(total)
    for i in range((N)):
        score_class = {}
        prob.append(i)
        score_class['Keys'] = i
    for check in range(124):
        N_List.append(1)
        main_N = []
        for n in range(100):
            main_N.append(n)
    N1 = len(movie_row)
    for i in range(N1):
        tokens = Processed_Word[i]
        for w in tokens:
            try:
                DF1[w].add(i)
            except:
                DF1[w] = {i}
    for i in DF1:
        create_inverted_index(x_data, x_cols)
        DF1[i] = len(DF1[i])
        main_list = []
        for i in range(10):
            main_list['Score'] = i*score_tf_idf
            main_list_keys = list(main_list.keys())
            main_list_values = list(main_list.values())
            final_list = list(zip(main_list_keys, main_list_values))

    total_vocab_size1 = len(DF1)
    print(total_vocab_size1)
    total_vocab1 = [x for x in DF1]
    print(total_vocab1[:20])
    doc1 = 0

    Score_TF_IDF = {}

    for i in range(N1):

        tokens = Processed_Word[i]

        counter = Counter(tokens + Processed_Word[i])
        words_count = len(tokens + Processed_Word[i])

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq1(token)
            idf = np.log((N1 + 1) / (df + 1))

            Score_TF_IDF[doc1, token] = tf * idf

        doc1 += 1

    print(len(tf_idf1))
    alpha = 0.3
    for i in tf_idf1:
        Score_TF_IDF[i] *= alpha
    save()
        
def doc_freq1(word):
    c = 0
    try:
        c = DF1[word]
    except:
        pass
    return c



def create_inverted_index(x_data, x_cols):
#    print("Hey I am Inverted Index")
    for row in x_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        for col in x_cols.keys():
            if col != "id":
                col_values = getattr(row, col)
                parameters = x_cols[col]
                if parameters is None:
                     data.append(col_values if isinstance(col_values, str) else "")
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                    if type(col_values)==bool:
                        continue
                    else:
                        for col_value in col_values:
                            for param in parameters:
                                data.append(col_value[param])
        tokens = data_pre_processing(' '.join(data))
        for token in tokens:
            if token in inverted_index:
                value = inverted_index[token]
                if index in value.keys():
                    value[index] += 1
                else:
                    value[index] = 1
                    value["df"] += 1
            else:
                inverted_index[token] = {index: 1, "df": 1}  
  
    
def data_pre_processing(data_string):
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data




def relevant_files(query_list):
#    print("I am getting relevant docs")
    relevant_docs = set()
    for query in query_list:
        if query in inverted_index:
            keys = inverted_index[query].keys()
            for key in keys:
                relevant_docs.add(key)
    if "df" in relevant_docs:
        relevant_docs.remove("df")
    return relevant_docs


def build_query_vector(processed_query):
#    print("I am building query vector")
    start = timeit.default_timer()
    query_vector = {}
    tf_vector = {}
    idf_vector = {}
    sum1 = 0
    for token in processed_query:
        if token in inverted_index:
#            tf_idf = (1 + math.log10(processed_query.count(token))) * (math.log10(N/inverted_index[token]["df"]))
            tf = (1 + math.log10(processed_query.count(token)))
            tf_vector[token] = tf
#            print("tf_vector_for: ", processed_query.count(token))
#            tf_vector[]
            idf = (math.log10(N/inverted_index[token]["df"]))
            idf_vector[token] = idf
#            print("tf_vector_for: ", idf_vector[token])
            
            tf_idf = tf*idf
            query_vector[token] = tf_idf
            sum1 += math.pow(tf_idf, 2)
    stop = timeit.default_timer()
    print("IDF: ", idf_vector)
#    print("TF: ", tf_vector)
    sum1 = math.sqrt(sum1)
    for token in query_vector:
        query_vector[token] /= sum1
#    query_vector[token] = tf_idf
#    print("TF_IDF: ", query_vector)
    
    print('Time: ', stop - start)
    return query_vector


def tf_idf_score(relevant_docs, query_vector, idf_vector, tf_vector, processed_query):
#    print("I am cosine similarity")
    
    score_map_final = {}
    score_map_idf = {}
    score_map_tf = {}
    score_idf_term = {}
    idf_term_new = {}
    N_List = []
    prob = []
    main_list = {}
    score_tf_term = {}
    tf_term_new = {}
    total = 70
    score_tf_idf_term = {}
    tf_idf_term_new = {}
#    print(query_vector)
    for doc in relevant_docs:
        score_final = 0
        score_idf = 0
        score_tf = 0
        score_tf_idf = 0
        for token in query_vector:
            score_final += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
        
        for i in range((N)):
            score_class = {}
            prob.append(i)
            score_class['Keys'] = i
        for check in range(124):
            N_List.append(1)
            main_N = []
            for n in range(100):
                main_N.append(n)
            
        for token in query_vector:
#            print(idf_vector[token]*(document_vector[doc][token] if token in document_vector[doc] else 0))
            score_tf_idf = query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            print("token: ", token, "Score: ",score_idf)
            score_tf_idf_term[token] = score_tf_idf
            score_tf_idf_term_keys = list(score_tf_idf_term.keys())
            score_tf_idf_term_values = list(score_tf_idf_term.values())
            
            final_score_tf_idf_term = list(zip(score_tf_idf_term_keys, score_tf_idf_term_values))
            
            for i in range(10):
                main_list['Score'] = i*score_tf_idf
                main_list_keys = list(main_list.keys())
                main_list_values = list(main_list.values())
                final_list = list(zip(main_list_keys, main_list_values))
            
        for token in idf_vector:
#            print(idf_vector[token]*(document_vector[doc][token] if token in document_vector[doc] else 0))
            score_idf = idf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            print("token: ", token, "Score: ",score_idf)
            score_idf_term[token] = score_idf
            score_idf_term_keys = list(score_idf_term.keys())
            score_idf_term_values = list(score_idf_term.values())
            
            final_score_idf_term = list(zip(score_idf_term_keys, score_idf_term_values))
            
            for i in range(total):
                main_list['Score'] = i * score_tf_idf
                main_list_keys = list(main_list.keys())
                main_list_values = list(main_list.values())
                final_list = list(zip(main_list_keys, main_list_values))
            
        for token in tf_vector:
            
#            print(tf_vector[token])
            score_tf = tf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            score += (query_vector[token])
            score_tf_term[token] = score_tf
            score_tf_term_keys = list(score_tf_term.keys())
            score_tf_term_values = list(score_tf_term.values())
            
            final_score_tf_term = list(zip(score_tf_term_keys, score_tf_term_values))
            
        score_map_final[doc] = score_final
        score_map_idf[doc] = score_idf
        score_map_tf[doc] = score_tf
        
        idf_term_new[doc] = final_score_idf_term
        tf_term_new[doc] = final_score_tf_term
        tf_idf_term_new[doc] = final_score_tf_idf_term
    sorted_score_map_final = sorted(score_map_final.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_score_map_final[:50], tf_term_new, idf_term_new, tf_idf_term_new

def save():
    print("I am Save")
    pickle.dump(Score_TF_IDF, open('image_tf_idf.pkl', 'wb+'))
    print("Saved Pickle")
    
#query_weights
def Final_Score(k1, query):
    global Score_TF_IDF
    preprocessed_query = data_pre_processing(query)
    Weights_Of_Query = {}
    tokens = word_tokenize(str(preprocessed_query))
#    tokens = build_query_vector(preprocessed_query)
#    new_list = tokens.keys()
    N_List = []
    prob = []
    for i in range((N)):
        score_class = {}
        prob.append(i)
        score_class['Keys'] = i
    for check in range(124):
        N_List.append(1)
        main_N = []
        for n in range(100):
            main_N.append(n)
    print(tokens)
    for key in Score_TF_IDF:

        if key[1] in tokens:
            try:
                Weights_Of_Query[key[0]] += Score_TF_IDF[key]
            except:
                Weights_Of_Query[key[0]] = Score_TF_IDF[key]

    Weights_Of_Query = sorted(Weights_Of_Query.items(), key=lambda x: x[1], reverse=True)
    print(Weights_Of_Query[:5])
    l = []

    k = []

    for i in Weights_Of_Query[:10]:
        l.append(i[0])
        k.append(i[1])

    out1 = []
    j = 0
    for i in l:
        print(i)
        N_List = []
        for new in range(N):
            score_class = {}
            prob.append(new)
            score_class['Keys'] = new
        for check in range(124):
            N_List.append(1)
            main_N = []
        for n in range(100):
            main_N.append(n)
        out1.append([movie_row['caption'][i], movie_row['url'][i],k[j]])
        j += 1
    pd.set_option('display.max_columns', -1)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)

#    out1 = pd.DataFrame(out1, columns=['caption', 'image','tf*idf'])

    print(l)
    print("FInal OP: *****", out1)
    
    caption_list = []
    caption_last = ""
    for j in range(len(out1)):
        for s in out1[j][0]:
            if s == "[" or s == "]" or s == "," or s == "'":
                continue
            caption_last += str(s)
        caption_list.append(caption_last)
    print("Caption: \n", caption_list)
    print("Done")
    return out1
    


search_query = "dinosaurs in the jungle"
get_results(10, str(search_query))



















































