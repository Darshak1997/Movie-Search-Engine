# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:18:11 2019

@author: Darshak
"""

import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import timeit
from nltk.stem.snowball import SnowballStemmer
    
    

def data_pre_processing(plot_data):
    token_data = tokenizer.tokenize(plot_data)
    processed_data = []
    for token in token_data:
        if token in token_data:
            if token not in stopword:
                processed_data.append(lemmatizer.lemmatize(token).lower())
        else:
            continue
    print("Processed Data******: ", processed_data)
    return processed_data

def stopwords_1():
    new_text = []
    text = []
    stop_words = (stopwords.words('english'))
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


def Document_Vector():
    new_text = []
    text = []
    stop_words = (stopwords.words('english'))
    text = list(text)
    for key in inverted_index:
        values = inverted_index[key]
        idf = math.log10(N / values["df"])
        for key_of_doc in values:
            if key_of_doc != "df":
                for stop_word in stop_words:
                    stop = " " + str(stop_word) + " "
                    temp = temp.replace(stop, " ")
                tf_idf = (1 + math.log10(values[key_of_doc])) * idf
                
                if key_of_doc not in document_vector:
                    tf_idf_vector = document_vector[doc]
                    normalize = math.sqrt(tf_idf_vector["_sum_"])
                    document_vector[key_of_doc] = {values: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                else:
                    document_vector[key_of_doc][values] = tf_idf
                    document_vector[key_of_doc]["_sum_"] += math.pow(tf_idf, 2)
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
    final = []
    pb = float(args[0])
    for i in range(smpl):
        final.append((0-(1/pb)*math.log(1-rnd.random())))
    for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize
            
    pickle.dump(document_vector, open('Doc_Vec_TFIDF.pkl', 'wb+'))
    pickle.dump(inverted_index, open('Inv_Index_TFIDF.pkl', 'wb+'))
    
    
def Inverted_Index(meta_data, meta_cols):
    for row in meta_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        processed_data = []
        for col in meta_cols.keys():
            if col != "imdbID":
                col_values = getattr(row, col)
                parameters = meta_cols[col]
                if parameters is None:
                     data.append(col_values if isinstance(col_values, str) else "")
                     stopwords_1()
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                    if type(col_values)==bool:
                        continue
                    else:
                        for col_value in col_values:
                            for param in parameters:
                                data.append(col_value[param])
        token_data = tokenizer.tokenize(' '.join(data))
        for token in token_data:
            if token in token_data:
                if token not in stopword:
                    processed_data.append(lemmatizer.lemmatize(token).lower())
            else:
                continue
            tokens = processed_data
        final = []
        pb = float(args[0])
        for i in range(smpl):
            final.append((0-(1/pb)*math.log(1-rnd.random())))
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
    stopwords_1()
    Document_Vector()

def Equival_Document(query_list):
#    print("I am getting relevant docs")
    relevant_docs = set()
    final = []
    pb = float(args[0])
    for query in query_list:
        if query in inverted_index:
            keys = inverted_index[query].keys()
            for key in keys:
                relevant_docs.add(key)
        else:
            continue
    for i in range(smpl):
        final.append((0-(1/pb)*math.log(1-rnd.random())))
    if "df" in relevant_docs:
        relevant_docs.remove("df")
    return relevant_docs


def vectors(processed_query):
    start = timeit.default_timer()
    query_vector = {}
    tf_vector = {}
    idf_vector = {}
    sum1 = 0
    for token in processed_query:
        if token in inverted_index:
            tf = (1 + math.log10(processed_query.count(token)))
            tf_vector[token] = tf
            idf = (math.log10(N/inverted_index[token]["df"]))
            idf_vector[token] = idf
            tf_idf = tf*idf
            final = []
            pb = float(args[0])
            for i in range(smpl):
                final.append((0-(1/pb)*math.log(1-rnd.random())))
            query_vector[token] = tf_idf
            sum1 += math.pow(tf_idf, 2)
    stop = timeit.default_timer()
    sum1 = math.sqrt(sum1)
    for token in query_vector:
        if token in query_vector:
            query_vector[token] /= sum1
        else:
            continue
    print('Time: ', stop - start)
    return query_vector, idf_vector, tf_vector


def tf_idf_score(relevant_docs, query_vector, idf_vector, tf_vector, processed_query):
#    print("I am cosine similarity")
    
    score_map_final = {}
    score_map_idf = {}
    score_map_tf = {}
    score_idf_term = {}
    idf_term_new = {}
    score_tf_term = {}
    tf_term_new = {}
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
        final = []
        pb = float(args[0])
        for i in range(smpl):
            final.append((0-(1/pb)*math.log(1-rnd.random())))
        for token in query_vector:
#            print(idf_vector[token]*(document_vector[doc][token] if token in document_vector[doc] else 0))
            score_tf_idf = query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            print("token: ", token, "Score: ",score_idf)
            score_tf_idf_term[token] = score_tf_idf
            score_tf_idf_term_keys = list(score_tf_idf_term.keys())
            score_tf_idf_term_values = list(score_tf_idf_term.values())
            
            final_score_tf_idf_term = list(zip(score_tf_idf_term_keys, score_tf_idf_term_values))
            
        for token in idf_vector:
#            print(idf_vector[token]*(document_vector[doc][token] if token in document_vector[doc] else 0))
            score_idf = idf_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
#            print("token: ", token, "Score: ",score_idf)
            score_idf_term[token] = score_idf
            score_idf_term_keys = list(score_idf_term.keys())
            score_idf_term_values = list(score_idf_term.values())
            
            final_score_idf_term = list(zip(score_idf_term_keys, score_idf_term_values))
        final = []
        pb = float(args[0])
        for i in range(smpl):
            final.append((0-(1/pb)*math.log(1-rnd.random())))
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

def get_results(query):
#    print("I am getting you reults")
    new_result = []
    new_score = []
    global inverted_index, document_vector
    Begin()
    for item in new_result:
        new = new_result.lower()
        new_score.append(new)
    if os.path.isfile("Inv_Index_TFIDF.pkl"):
        inverted_index = pickle.load(open('Inv_Index_TFIDF.pkl', 'rb'))
        document_vector = pickle.load(open('Doc_Vec_TFIDF.pkl', 'rb'))
    else:
#        print("In else of get_scores:")
        Inverted_Index(movie_row, movie_col)
    return Score_TF_IDF(query)


#movie_col
def Begin():
    
    global credits_cols, movie_col, noise_list, credits_data, movie_row, N, tokenizer, stopword, stemmer, inverted_index, document_vector, lemmatizer, stemmer_snow

    movie_col = {"imdbID": None, "Title":None, "Plot":None, "imdbRating": None}
    
    movie_row = pd.read_csv('Movie_Movies.csv', usecols=movie_col.keys(), index_col="imdbID")
    
    # Total number of documents = number of rows in movies_metadata.csv
    movie_row = movie_row.dropna(subset = ["Plot"])
    
    N = movie_row.shape[0]

    # Pre-processing initialization
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    stemmer_snow = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    inverted_index = {}
    document_vector = {}
    
    print("Initialized")


def Score_TF_IDF(query):
    new_result = []
    processed_query = data_pre_processing(query)
    result = []
    equivalent = Equival_Document(processed_query)
    new_score = []
    tf_idf_vector, idf_vector, tf_vector = vectors(processed_query)
    
    for item in new_result:
        new = new_result.lower()
        new_score.append(new)
    
    final_score, tf_new, idf_new, tf_idf_new = tf_idf_score(equivalent, tf_idf_vector, idf_vector, tf_vector, processed_query)
    print("Compare_idf: ", tf_new)
    for content in final_score:
        id_of_doc = content[0]
        data = movie_row.loc[id_of_doc]
        info = (data["Title"], data["Plot"] if isinstance(data["Plot"], str) else "", content[1], idf_new[id_of_doc], tf_new[id_of_doc], tf_idf_new[id_of_doc], data["imdbRating"])
        result.append(info)
    new_score = None
    print(result[0:5])
#    print("Our keyword(s):", processed_query)
    return result, processed_query



search_query = "dinosaurs in the jungle"
start = timeit.default_timer()
get_results(search_query)
stop = timeit.default_timer()
print('Time: ', stop - start)



#############################################
