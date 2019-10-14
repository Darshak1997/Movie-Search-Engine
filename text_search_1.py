# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:18:11 2019

@author: Darshak
"""

import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

def create_inverted_index(x_data, x_cols):
#    print("Hey I am Inverted Index")
    for row in x_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        for col in x_cols.keys():
            if col != "imdbID":
                col_values = getattr(row, col)
#                print(col_values)
                parameters = x_cols[col]
                if parameters is None:
                     data.append(col_values if isinstance(col_values, str) else "")
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
#                    print(col_values)
#                    break
                    if type(col_values)==bool:
                        continue
                    else:
                        for col_value in col_values:
#                            print(col_value)
                            for param in parameters:
                                data.append(col_value[param])
#        insert(index, pre_processing(' '.join(data)))
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
#    print(inverted_index)
    stopwords_1()
                
                
def data_pre_processing(data_string):
#    for noise in noise_list:
#        data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(lemmatizer.lemmatize(t).lower())
    return processed_data

def stopwords_1():
    new_text = []
    stop_words = (stopwords.words('english'))
    text = []
    text = list(text)
    for item in text:
        if type(item) == str:
            temp = item.lower()
            temp = temp.replace("[^a-zA-Z]", " ").replace("\'", "").replace(',', "").replace('.', "").replace('é', 'e').replace('?', '').replace('!', '')
            for stop_word in stop_words:
                stop = " " + str(stop_word) + " "
                temp = temp.replace(stop, " ")
            new_text.append(temp)
        else:
            new_text.append('Nan')
    text = None
    return text

def build_doc_vector():
#    print("I am Build_doc_vector")
    for token_key in inverted_index:
        token_values = inverted_index[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                tf_idf = (1 + math.log10(token_values[doc_key])) * idf
                
                if doc_key not in document_vector:
                    document_vector[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
#                    document_vector[doc_key] = {token_key: tf_idf, "_sum_": (tf_idf)}
                else:
                    document_vector[doc_key][token_key] = tf_idf
                    document_vector[doc_key]["_sum_"] += math.pow(tf_idf, 2)
#                    document_vector[doc_key]["_sum_"] += tf_idf
    
    for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize

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
    query_vector = {}
    tf_vector = {}
    idf_vector = {}
    sum1 = 0
    for token in processed_query:
        if token in inverted_index:
#            tf_idf = (1 + math.log10(processed_query.count(token))) * (math.log10(N/inverted_index[token]["df"]))
            tf = (1 + math.log10(processed_query.count(token)))
            tf_vector[token] = tf
#            tf_vector[]
            idf = (math.log10(N/inverted_index[token]["df"]))
            idf_vector[token] = idf
            tf_idf = tf*idf
            query_vector[token] = tf_idf
            sum1 += math.pow(tf_idf, 2)
#            sum1 += tf_idf
    print("IDF: ", idf_vector)
#    print("TF_IDF: ", query_vector)
    sum1 = math.sqrt(sum1)
#    sum1 = sum(query_vector[token])
    for token in query_vector:
        query_vector[token] /= sum1
#    print(query_vector[token] * document_vector[doc][token])[:10]
    print("TF_IDF: ", query_vector)
    return query_vector, idf_vector, tf_vector
    

def cosine_similarity(relevant_docs, query_vector, idf_vector, tf_vector):
#    print("I am cosine similarity")
    score_map_final = {}
    score_map_idf = {}
    score_map_tf = {}
    idf_list = []
#    print(query_vector)
    for doc in relevant_docs:
        score_final = 0
        score_idf = 0
        score_tf = 0
        for token in query_vector:
            score_final += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
        for token in idf_vector:
            score_idf += idf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            score_map_idf[token] = score_idf
            idf_list.append(score_idf)
        for token in tf_vector:
            score_tf = tf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            score += (query_vector[token])
        score_map_final[doc] = score_final
        score_map_idf[doc] = score_idf
        score_map_tf[doc] = score_tf
    sorted_score_map_final = sorted(score_map_final.items(), key=operator.itemgetter(1), reverse=True)
    sorted_score_map_idf = sorted(score_map_idf.items(), key=operator.itemgetter(1), reverse=True)
    sorted_score_map_tf = sorted(score_map_tf.items(), key=operator.itemgetter(1), reverse=True)
#    print(sorted_score_map[:10])
    print("Sorted Score_idf: ",sorted_score_map_idf[:5])
    print("Sorted Score_tf: ",sorted_score_map_tf[:5])
#    print("idf_list: ", idf_list[:10])
    return sorted_score_map_final[:50], sorted_score_map_idf[:50], sorted_score_map_tf[:50]

def get_results(query):
#    print("I am getting you reults")
    global inverted_index, document_vector
    initialize()
    if os.path.isfile("invertedIndexPickle.pkl"):
        inverted_index = pickle.load(open('invertedIndexPickle.pkl', 'rb'))
        document_vector = pickle.load(open('documentVectorPickle.pkl', 'rb'))
    else:
        print("In else of get_scores:")
        build()
        save()
    return eval_score(query)

def initialize():
    global data_folder, credits_cols, movie_col, noise_list, credits_data, movie_row, N, tokenizer, stopword, stemmer, inverted_index, document_vector, lemmatizer 

    # Data configurations
    #data_folder = '/home/npandya/mysite/data/'
    data_folder = 'F:/Data Mining/Assignments/Assignment 1/'
#    credits_cols = {"id": None, "cast":['character', 'name'], "crew":['name']}
#    meta_cols = {"id": None, "genres":['name'], "original_title":None, "overview":None,"poster_path":None,
#                     "production_companies":['name'], "tagline":None}
    movie_col = {"imdbID": None, "Title":None, "Plot":None, "imdbRating": None}

    # Read data
#    credits_data = pd.read_csv(data_folder +'credits.csv', usecols=credits_cols.keys(), index_col="id")
#    meta_data = pd.read_csv(data_folder + 'movies_metadata.csv', usecols=meta_cols.keys(), index_col="id")
    
    movie_row = pd.read_csv(data_folder + 'Movie_movies.csv', usecols=movie_col.keys(), index_col="imdbID")
    # Total number of documents = number of rows in movies_metadata.csv
    movie_row = movie_row.dropna(subset = ["Plot"])
    N = movie_row.shape[0]

    # Pre-processing initialization
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    inverted_index = {}
    document_vector = {}
    print("Initialized")

def build():
#    print("I am Build")
#    print("Creating inverted index for credits data...")
#    create_inverted_index(credits_data, credits_cols)
    print("Creating inverted index for meta data...")
    create_inverted_index(movie_row, movie_col)
    print("Building doc vector...")
    build_doc_vector()
    print("Built index and doc vector")

def save():
#    print("I am Save")
    pickle.dump(inverted_index, open('invertedIndexPickle.pkl', 'wb+'))
    pickle.dump(document_vector, open('documentVectorPickle.pkl', 'wb+'))
    print("Saved both")

def eval_score(query):
#    print("I am Evaluating score")
    result = []
    
    processed_query = data_pre_processing(query)
    print(processed_query)
    
    relevant = relevant_files(processed_query)
    
    query_vector, idf_vector, tf_vector = build_query_vector(processed_query)
    
    new_result = []
    
    new_score = []
    
    for item in new_result:
        new = new_result.lower()
        new_score.append(new)
    
    sorted_score_list_final, sorted_score_list_idf, sorted_score_list_tf = cosine_similarity(relevant, query_vector, idf_vector, tf_vector)
#    sorted_score_list = cosine_similarity(relevant, query_vector)
#    print("final_final_eval_score: ", (sorted_score_list_idf[:5]))
#    addition = sorted_score_list_final + sorted_score_list_idf + sorted_score_list_tf
#    print("Addition is: ",addition[:5])
    for entry in sorted_score_list_final:
##        print("hello")
        doc_id = entry[0]
##        print(entry[0])
        row = movie_row.loc[doc_id]
#        print(row)
        info = (row["Title"], row["Plot"] if isinstance(row["Plot"], str) else "", entry[1],row["imdbRating"])
        result.append(info)
    print(result[0][1])
#    import colorama
#    from colorama import Fore, Back, Style
#
#    colorama.init()
#    for i in processed_query:
#        if i in result[0][1]:
#            print(Back.GREEN + i + Style.RESET_ALL)
#            new = Back.GREEN + i + Style.RESET_ALL
#        result[i][1].replace(new)
#            print(CRED + i + CEND)
#    for entry in sorted_score_list_final:
#        for entry in sorted_score_list_idf:
#            for entry in sorted_score_list_idf:
##        print("hello")
#                doc_id = entry[0]
##        print(entry[0])
#                row = movie_row.loc[doc_id]
##                print(row)
#                info = (row["Title"], row["Plot"] if isinstance(row["Plot"], str) else "", entry[1],row["imdbRating"])
#                result.append(info)
    new_score = None
#    print(result[0:5])
    return result, processed_query

#def get_movie_info(sorted_score_list):
##    print("I am Getting movie info")
#    result = []
#    for entry in sorted_score_list:
#        doc_id = entry[0]
##        print(type(doc_id))
##        if type(doc_id) == str:
#        row = movie_row.loc[doc_id]
#        info = (row["Poster"], row["Title"],
#                row["Plot"] if isinstance(row["Plot"], str) else "",
#                entry[1],
#                row["imdbRating"])
##        else:
##            continue
#        result.append(info)
#    
##    print(result[0:5])
#    return result
#

#search_query = "Kid alone at home"
#get_results(search_query)



#############################################