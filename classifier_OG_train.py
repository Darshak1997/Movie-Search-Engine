# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:12:16 2019

@author: Darshak
"""

import pandas as pd
import os, pickle, ast, operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer



# To calculate the final score from prior probability
# And the posterior probability and the store it in a list
def Score_Genre_Prob(query):
    print("im eval")
    genre_score = {}
    mainList = []
    # Cleaning the genre column
    unwanted_list = ['Vision View Entertainment', 'Carousel Productions', 'Pulser Productions', 'Sentai Filmworks', 'Aniplex', 'Mardock Scramble Production Committee', 'Odyssey Media', 'GoHands', 'Rogue State', 'Telescene Film Group Productions', 'BROSTA TV', 'The Cartel']
    # Pre process the input query
    processed_query = pre_processing(query)
    prob = []
    for i in range(len(data_test)):
        score_class = {}
        prob.append(i)
        score_class['Keys'] = i
    Score_New()
    # Calculating the final score based on Genre class
    for genre in Prob_Prior.keys():
        if genre in unwanted_list:
            continue
        score = Prob_Prior[genre]
        # Call this function to store new values
        Score_New()
        # Calculate the score of the token in the processed query
        # To generate the final score
        for token in processed_query:
            if (genre, token) in Prob_Post.keys():
                score *= Prob_Post[(genre, token)]
        genre_score[genre] = score
    # Calculated the probability score and their equivalent Genre Class
    Result_Sorted = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
#    eval_result_test(query)
    prob_score = []
    print(Result_Sorted[0][0])
    for i in Result_Sorted:
        prob_score.append(i[1])
    Total_prob = sum(prob_score)
    prob_score_new = []
    # To Calculate the percentages of the Genre class
    for indiv in prob_score:
        prob_score_new.append((indiv/Total_prob)*100)
    print(type(Result_Sorted))
    
    return Result_Sorted[:5], prob_score_new[:5]

# This is the same function as above to just test our training set
def Score_Genre_Prob_test(query):
#    print("im eval test")
    genre_score = {}
    # Cleaning the genre column
    unwanted_list = ['Vision View Entertainment', 'Carousel Productions', 'Pulser Productions', 'Sentai Filmworks', 'Aniplex', 'Mardock Scramble Production Committee', 'Odyssey Media', 'GoHands', 'Rogue State', 'Telescene Film Group Productions', 'BROSTA TV', 'The Cartel']
    # Pre process the input query
    processed_query = pre_processing(query)
    prob = []
    for i in range(len(data_test)):
        score_class = {}
        prob.append(i)
        score_class['Keys'] = i
    Score_New()
    for genre in Prob_Prior.keys():
        if genre in unwanted_list:
            continue
        score = Prob_Prior[genre]
        Score_New()
        # print("For genre: ", genre, ", prior score: ", score)
        for token in processed_query:
            if (genre, token) in Prob_Post.keys():
                score *= Prob_Post[(genre, token)]
                # print("token: ", token, ", score: ", score)
        genre_score[genre] = score
    Result_Sorted = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
    predicted_compare = [Result_Sorted[0][0], Result_Sorted[1][0], Result_Sorted[2][0]]
#    print(predicted_compare)
    return predicted_compare

# Storing The score of the new value
def Score_New():
    prob = []
    N_List = []
    for i in range(len(data_test)):
        score_class = {}
        prob.append(i)
        score_class['Keys'] = i
    for check in range(124):
        N_List.append(1)
        main_N = []
        for n in range(100):
            main_N.append(n)
            accuracy(Prob_Post, Prob_Prior)
              
    

    
# Use this Function to train our Training Set and 
# Calculate which token belongs to which Genre class
# And also Calculate the prior and the posterior Probabilities for 
# Each token in that class.
def Calculate_Token_Prob():
    print("im build and save")
    Row = 0
    token_count = 0
    Prob_Post = {}
    score_predicted = {}
    Count_Token = {}
    genre_count_map = {}
    N_List = []
    for row in data_train.itertuples():
        Processed_Word = []
        genres = []
        for keys in score_predicted.keys():
            keys += 1
        for col in meta_cols.keys():
            col_values = getattr(row, col)
            parameters = meta_cols[col]
            for scores in score_predicted.values():
                print(score_predicted.values())
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            for check in range(N):
                N_List.append(N)
                main_N = []
                for n in N_List:
                    main_N.append(n)
                    main_N.pop()
                    
            if parameters is None:
                Processed_Word.append(col_values if isinstance(col_values, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            
            else:
                for i in range(10):
                    score_predicted.keys(i)
                col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                for col_value in col_values:
                    for param in parameters:
                        genres.append(col_value[param])

        tokens = pre_processing(' '.join(Processed_Word))
        for genre in genres:
            if genre in genre_count_map:
                genre_count_map[genre] += 1
            else:
                genre_count_map[genre] = 1
            for token in tokens:
                token_count += 1
                if (genre, token) in Count_Token:
                    Count_Token[(genre, token)] += 1
                else:
                    Count_Token[(genre, token)] = 1

        Row += 1
    for (genre, token) in Count_Token:
        Prob_Post[(genre, token)] = Count_Token[(genre, token)] / token_count

    Prob_Prior = {x: genre_count_map[x]/Row for x in genre_count_map}
    
    accuracy(Prob_Post, Prob_Prior)
    
    # Storing the Prior and Posterior Probability Scores in a Pickle File
    pickle.dump(Prob_Prior, open('classifierPicklePrior.pkl', 'wb+'))
    pickle.dump(Prob_Post, open('classifierPicklePost.pkl', 'wb+'))
    
    return (Prob_Prior, Prob_Post)

# Use this function to input the plot of the Test set
# in the training set to compare the predicted Genre
# And the Actual Genre
def Calculate_Token_Prob_test():
    print("im build and save test")
    genre_count_map_actual = {}
    token_genre_count_map_actual = {}
    count = 0
    for row in data_test.itertuples():
        keywords_actual = []
        genres_actual = []
        score_predicted = {}
        for keys in score_predicted.keys():
            keys += 1
        for col in meta_cols.keys():
            col_values = getattr(row, col)
            parameters = meta_cols[col]
            prob = []
            for i in range(len(data_test)):
                score_class = {}
                prob.append(i)
                score_class['Keys'] = i
            for scores in score_predicted.values():
                print(score_predicted.values())
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            if parameters is None:
                keywords_actual.append(col_values if isinstance(col_values, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            else:
                for i in range(10):
                    score_predicted.keys(i)
                col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                for col_value in col_values:
                    for param in parameters:
                        genres_actual.append(col_value[param])
            prob = []
            for i in range(len(data_test)):
                score_class = {}
                prob.append(i)
                score_class['Keys'] = i
        for genre in genres_actual:
            if genre in genre_count_map_actual:
                genre_count_map_actual[genre] += 1
            else:
                genre_count_map_actual[genre] = 1
                prob = []
                for i in range(len(data_test)):
                    score_class = {}
                    prob.append(i)
                    score_class['Keys'] = i
                Score_New()
            for keyword in keywords_actual:
                if (genre, keyword) in token_genre_count_map_actual:
                    token_genre_count_map_actual[(genre, keyword)] += 1
                else:
                    token_genre_count_map_actual[(genre, keyword)] = 1
    
    token_genre = []
    genre_list = []
    overview_list = []
    
    for key in token_genre_count_map_actual.keys():
        token_genre.append(key)
    
    for i in token_genre:
        if i[1] not in overview_list:
            overview_list.append(i[1])
            predicted_compare = Score_Genre_Prob_test(i[1])
#            print("Predcited****: ", predicted_compare)
#            print("Actual******: ", i[0])
            if i[0] in predicted_compare:
                count += 1
    
    print("Accuracy: ", count/13640)

# Tokenize the plot and the input Query
# Also stem the data
def pre_processing(data_string):
    #print("im pre processing")
    # for noise in noise_list:
    #     data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer_snow.stem(t).lower())
    return processed_data

    
# Splitting the dataset in 70:30 ratio
def train_test_split1(meta_data):
        x = meta_data
        #y = self.MovieData['overview']
        x_train, x_test = train_test_split(x,test_size = 0.3, random_state=4)
        print(len(x_train))
        return x_train, x_test
    
# Where all the functions are accessed from.
# If we already have a pickle file then just calculate the result.
def Final_Score(query):
    print("im get results")
    initialize()
    global Prob_Prior
    global Prob_Post
    # If pickle has already been created then just calculate the result
    if os.path.isfile("classifierPicklePrior.pkl"):
        Prob_Post = pickle.load(open('classifierPicklePost.pkl', 'rb'))
        Prob_Prior = pickle.load(open('classifierPicklePrior.pkl', 'rb'))
    # If you are ruuning this code for the first time then it would go in this 
    else:
        # To just run the trained model
        (Prob_Prior, Prob_Post) = Calculate_Token_Prob()
        # To validate and calculate the accuracy of the training set call the test function here
        # Add your Function here 
    return Score_Genre_Prob(query)
    
# Calculating the Accuracy of our model
# by comparing the actual Genre of the plot in the test set
# with the predicted genre from the training set
def accuracy(post_probability, post_probability_test):
#    print(post_probability_test.keys())
    count = 0
    for keys in post_probability_test.keys():
        if keys in Prob_Post.keys():
            print(keys)
            count += 1
#    print("Count*****: ", count)
#    print("Length of dict******: ", len(post_probability_test))
    accuracy_score = count/len(post_probability_test)
#    print("Accuracy*****: ", accuracy_score)
#    for (genre, token) in post_probability:
#        print(genre)
#        train_no_prob[(token)] = train_no_prob[(genre)]
#    print(train_no_prob)
    

def initialize():
    print("I,m Initialized")
    global meta_data, meta_cols, tokenizer, stopword, stemmer, data_train, data_test, probabilities_prior_test, probabilities_post_test, post_probability, stemmer_snow, N
    # Select Which columns to select from the dataset
    meta_cols = {"genres":['name'], "overview":None, "tagline":None}
    # Read the Dataset and Select columns as initialized above
    meta_data = pd.read_csv('movies_metadata.csv', usecols=meta_cols.keys())
    N = meta_data.shape[0]
    print(N)
    #Split the dataset by going in this function
    data_train, data_test = train_test_split1(meta_data)
    # Initialize the preprocessing variables
    # Tokenize all the words in the range below
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # Get all the stopwrods that we want to remove
    stopword = stopwords.words('english')
#    stemmer = PorterStemmer()
    # Stem all the tokenized words
    stemmer_snow = SnowballStemmer("english")
    

  
