# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

# Purpose: For cleaning data & build a recommender system

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords #pip install nltk 
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load the CSV file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/sgcarmart_usedcar_info.csv')
#df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/sgcarmart_usedcar_info.csv')
    
#preview the csv file
print(df.describe())
print(df.info())
print(df.head())


#Notes for report
#Content Based Filtering, match users directly to product based on what they view and is similiar
#use neighbourhood approach
#A few recommender to use
#1.) Popularity based: Item with lots of listing are what SGCARMART can try to push
#2.) Content Based Filtering for those item which are similiar to what u view
#3.) Interface tor ecommend based on your set budget etc #Will need profile learner

#Features of product
#We have standard features here
#Kwan Yick Code for all features, column A to Q

#Clean data, remove all the Null
df_nonull = df.dropna()

# Print the number of rows before and after dropping nulls
print('Number of rows before dropping nulls:', len(df))
print('Number of rows after dropping nulls:', len(df_nonull))

#Create data frame with selected attributes
features = df_nonull[['price','depreciation']]
print(features.describe())
print(features.info())
print(features.head())

#Remove punctuation
features_cleaned = features.copy()
features_cleaned['price']= features_cleaned['price'].str.replace('[\$,]', '', regex=True)
features_cleaned['price']= features_cleaned['price'].str.replace(',', '', regex=True)
features_cleaned['depreciation']= features_cleaned['depreciation'].str.replace('[\$,]', '', regex=True)
features_cleaned['depreciation']= features_cleaned['depreciation'].str.replace(',', '', regex=True)

#Remove N.A. 
# Replace "N.A." values with NaN
features_cleaned.replace('N.A', pd.NA, inplace=True)
# Calculate the median value for each column
median_values = features_cleaned.median()
# Fill NaN values with median value for each column
features_cleaned.fillna(median_values, inplace=True)

features_cleaned.sort_values('price', ascending=False, inplace=True)
print(features_cleaned.head())

#Normalize data to avoid bias towards any attribute
#features_normalized = (features_cleaned - features_cleaned.mean()) / features_cleaned.std()
#print(features_normalized.head())
# Error in normalizing data; need to check further

# Compute the pairwise cosine similarity between the items
#item_similarities = cosine_similarity(features_cleaned.T)

# Get the top k most similar items for each item
#k = 5
#top_k_similar_items = item_similarities.argsort()[:, :-k-1:-1]

# Define a function to get the top k recommended items for a given item
#def get_top_k_recommendations(item_id):
#    top_k_items = top_k_similar_items[item_id]
#    top_k_items_scores = item_similarities[item_id][top_k_items]
#    top_k_items_df = features_cleaned.iloc[top_k_items][['price','mileage']]
#    top_k_items_df['score'] = top_k_items_scores
#    top_k_items_df = top_k_items_df.sort_values(by='score', ascending=False)
#    return top_k_items_df.head(k)




#Ivan Code for column R, S , P 

#Converting features to string
df['car_features'] = df['car_features'].astype(str)
df['car_accessories'] = df['car_accessories'].astype(str)
df['descriptions'] = df['descriptions'].astype(str)


# define the columns to clean
cols_to_clean = ['car_features', 'car_accessories', 'descriptions']

# convert the specified columns to lowercase
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.str.lower())

# remove numbers and unwanted characters using regular expressions
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.str.replace(r'[^a-z\s]', '', regex=True))

# tokenize the specified columns into words
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.str.split())

# remove stop words using nltk library
stop_words = set(stopwords.words('english'))
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.apply(lambda y: [word for word in y if not pd.isnull(word) and word not in stop_words]))

# lemmatize the words using nltk library
lemmatizer = WordNetLemmatizer()
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.apply(lambda y: [lemmatizer.lemmatize(word) for word in y]))

# join the words back into a single string
df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.apply(lambda y: ' '.join(y)))

# fill NaN values with empty string
#df[cols_to_clean] = df[cols_to_clean].replace(np.nan, '', regex=True)

# vectorize the cleaned up columns using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[cols_to_clean].apply(lambda x: ' '.join(x), axis=1))

# print the resulting feature names and document-term matrix
print(vectorizer.get_feature_names())
print(X.toarray())


# create a dataframe from the resulting document-term matrix
tfidf_df = pd.DataFrame(X.toarray(),
                        columns=vectorizer.get_feature_names())
tfidf_df.index = df['model']

print(tfidf_df)

# calculate cosine similarity between each pair of rows in the tfidf_df dataframe
cos_sim = cosine_similarity(tfidf_df)

# create a new dataframe with the cosine similarity scores
cos_sim_df = pd.DataFrame(cos_sim, columns=tfidf_df.index, index=tfidf_df.index)

# display the top 10 most similar models to a given model (in this example, model 'Honda Vezel')
model_name = 'Honda Vezel 1.5A X'
top_similar_models = cos_sim_df[[model_name]].sort_values(by='model', ascending=False)[1:6]

print(top_similar_models)








  


