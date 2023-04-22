# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:41:56 2023

@author: kwanyick, Ivan
"""

# Purpose: For cleaning data & build a recommender system
#### TO INCLUDE FEATURE ENGINEERING before finalizing RCS

##Start##

##Data Cleaning##

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords #pip install nltk 
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import datetime

#Load the CSV file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
#df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
    
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
#3.) Interface to recommend based on your set budget etc #Will need profile learner

#Features of product
#We have standard features here
#Kwan Yick Code for all features, column A to Q

#Clean data, remove all the Null
df_nonull = df.dropna()

# Print the number of rows before and after dropping nulls
print('Number of rows before dropping nulls:', len(df))
print('Number of rows after dropping nulls:', len(df_nonull))
print(df_nonull.head())

#Create data frame with unique ID(new column) and selected attributes
# Generate a sequence of integers for the unique IDs
ids = range(1, len(df_nonull)+1)
df_nonull['car_id']= ids
col = df_nonull.pop('car_id')
df_nonull.insert(0,'car_id',col)
print(df_nonull.head())
print(df_nonull.describe())
print(df_nonull.info())  


df = df_nonull
print(df.head())
print(df.info())

df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].replace('N.A', np.nan)
df['price'] = df['price'].astype(float)

#Building a sample recommender system for testing
#1 Using just the features price

price_df = df[['car_id', 'model', 'price']]
price_df['price'] = price_df.groupby('model')['price'].transform(lambda x: x.fillna(x.mean()))
price_df = price_df.dropna(subset=['price'])
null_counts = price_df.isnull().sum()
print(null_counts)
print(price_df.head())
print(price_df.info())

# Define the Reader object to read the data
reader = Reader(rating_scale=(price_df['price'].min(), price_df['price'].max()))

# Load the data from the dataframe into the Surprise Dataset object
data = Dataset.load_from_df(price_df[['model', 'car_id', 'price']], reader=reader)

# Define the SVD algorithm and fit the training data
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Define a function to get recommended products
def get_recommendations(price):
    # Create a list of items that the user has not yet rated
    model_list = price_df[price_df['price'] <= price]['model'].unique().tolist()
    all_models = price_df['model'].unique().tolist()
    unrated_models = list(set(all_models) - set(model_list))
    unrated_ids = price_df[price_df['model'].isin(unrated_models)]['car_id'].unique().tolist()

    # Create a list of tuples containing the unrated items and their predicted ratings
    predictions = []
    for car_id in unrated_ids:
        prediction = algo.predict(uid='user', iid=car_id)
        predictions.append((prediction.iid, prediction.est))

    # Sort the predictions by rating (highest first) and return the top 5
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:5]

    # Map the item IDs to their corresponding titles and return the result
    recommendations = []
    for car_id, rating in top_predictions:
        model = price_df[price_df['car_id'] == car_id]['model'].values[0]
        recommendations.append((model, rating))
    return recommendations

# Example usage given budget of 50000
price = 50000
recommendations = get_recommendations(price)
print(f"Recommended models within price range of {price}:")
for rec in recommendations:
    print(f"Model: {rec[0]}, Predicted Rating: {rec[1]}")
    
#Evaluating the recommender 1

# Split the data into a training set and a test set
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define the SVD algorithm
algo = SVD()

# Perform k-fold cross-validation on the training set
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

# Print the mean RMSE and MAE scores across all folds
print(f"Mean RMSE: {results['test_rmse'].mean():.3f}")
print(f"Mean MAE: {results['test_mae'].mean():.3f}")

#The output means that the mean RMSE (Root Mean Squared Error) of the model is 2090304.553, and the mean MAE (Mean Absolute Error) is 1791902.207.
#The RMSE is a measure of the difference between the predicted values and the actual values. The lower the RMSE value, the better the model's performance. In this case, the RMSE value is quite high, indicating that the model may not be performing well.
#The MAE is another measure of the difference between the predicted values and the actual values. It is the average of the absolute differences between the predicted values and the actual values. The lower the MAE value, the better the model's performance. In this case, the MAE value is also quite high, indicating that the model may not be performing well.
#Overall, these evaluation metrics suggest that the model may need further improvement or fine-tuning to achieve better performance.

#after initial recommender 1, it seems that an extensive data cleaning is require. Moving to data cleaning first
#considering cleaned and rolled up data here

## Adding latest cleaned and rolled up data here ## 
#df_alllesstext = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
df_alllesstext = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')