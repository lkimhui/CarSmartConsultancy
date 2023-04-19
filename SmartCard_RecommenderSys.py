# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

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
import tkinter as tk


#after initial recommender 1, it seems that an extensive data cleaning is require. Moving to data cleaning first
#considering cleaned and rolled up data here

## Adding latest cleaned and rolled up data here ## 
#df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')


##5 Recommender in total ##

#1.) basic, just price range (done)

#Load the CSV file
df_old = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
#df_old = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
    
#preview the csv file
print(df.describe())
print(df.info())
print(df.head())

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


#2.) Similarity based on product text descritpion columns (done)

#Mining the text data
#1 Using just the words description to build a recommdner

#Converting features to string
df_alllesstext['car_features'] = df_alllesstext['car_features'].astype(str)
df_alllesstext['car_accessories'] = df_alllesstext['car_accessories'].astype(str)
df_alllesstext['descriptions'] = df_alllesstext['descriptions'].astype(str)


# define the columns to clean
cols_to_clean = ['car_features', 'car_accessories', 'descriptions']

# convert the specified columns to lowercase
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.str.lower())

# remove numbers and unwanted characters using regular expressions
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.str.replace(r'[^a-z\s]', '', regex=True))

# tokenize the specified columns into words
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.str.split())

# remove stop words using nltk library
stop_words = set(stopwords.words('english'))
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.apply(lambda y: [word for word in y if not pd.isnull(word) and word not in stop_words]))

# lemmatize the words using nltk library
lemmatizer = WordNetLemmatizer()
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.apply(lambda y: [lemmatizer.lemmatize(word) for word in y]))

# join the words back into a single string
df_alllesstext[cols_to_clean] = df_alllesstext[cols_to_clean].apply(lambda x: x.apply(lambda y: ' '.join(y)))

# fill NaN values with empty string
#df[cols_to_clean] = df[cols_to_clean].replace(np.nan, '', regex=True)

# vectorize the cleaned up columns using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_alllesstext[cols_to_clean].apply(lambda x: ' '.join(x), axis=1))

# print the resulting feature names and document-term matrix
print(vectorizer.get_feature_names())
print(X.toarray())


# create a dataframe from the resulting document-term matrix
tfidf_df = pd.DataFrame(X.toarray(),
                        columns=vectorizer.get_feature_names())
tfidf_df.index = df_alllesstext['model']

print(tfidf_df)

# calculate cosine similarity between each pair of rows in the tfidf_df dataframe
cos_sim = cosine_similarity(tfidf_df)

# create a new dataframe with the cosine similarity scores
cos_sim_df = pd.DataFrame(cos_sim, columns=tfidf_df.index, index=tfidf_df.index)

# display the top 10 most similar models to a given model (in this example, model 'Honda Vezel')
model_name = 'Volvo V60 T5 R-Design'
top_similar_models = cos_sim_df[[model_name]].sort_values(by=model_name, ascending=False)[1:6]

print(top_similar_models)

print(df_alllesstext.head())

#3.) base on all non text features


#4.) base on combine feature, text and non text (This is for user who dunno what they want)


# 5.) based on user input, price, car type and age (This is for user who roughly know what they want)

df_5 = df
print(df_5.head())

# Higher Safety Rating 
# Reasoning: 
    # Car Type (SUV,  Pickup Truck) has Better safety profile; generally safer in collisions than smaller cars.
    # Newer cars usally have better safety features and technology

suv_pickup_cars = df_5[(df_5['types_SUV'] == 1) | (df_5['types_Truck'] == 1)]

top_50_cars_safety = suv_pickup_cars.sort_values(['age_of_car'], ascending=[True]).head(50)
print(top_50_cars_safety.head())

# Higher Comfort Rating 
# Reasoning: 
    # Transmission (Automatic)
    # Car Type (Luxury) may give more advanced comfort features.
    # Newer launched cars usally have  newer cars may have better suspension, quieter cabins, and more advanced technology

luxury_auto_cars = df_5[(df_5['types_Luxury Sedan'] == 1) & (df_5['transmission_Auto'] == 1)]

top_50_cars_comfort = luxury_auto_cars.sort_values(['years_since_launch'], ascending=[True]).head(50)
print(top_50_cars_comfort.head())

#Define user input 


# Define function for High Safety Rating 
def high_safety(user_input):
    
    if user_input == 'safety':
        suv_pickup_cars = df_5[(df_5['types_SUV'] == 1) | (df_5['types_Truck'] == 1)]
        top_50_cars_safety = suv_pickup_cars.sort_values(['age_of_car'], ascending=[True]).head(50)
        output = print(top_50_cars_safety.head())
    
    else:
        luxury_auto_cars = df_5[(df_5['types_Luxury Sedan'] == 1) & (df_5['transmission_Auto'] == 1)]
        top_50_cars_comfort = luxury_auto_cars.sort_values(['years_since_launch'], ascending=[True]).head(50)
        output = print(top_50_cars_comfort.head())   
    
    return output

# Attempt getting user input using Tkinter (a Python GUI toolkit)



# Define the function to get user input
def get_input():
    # Get the value of the entry widget
    user_input = entry.get()
    label.config(text="You entered: " + user_input)
    
    # Create a new DataFrame with the user input
    new_df_5 = pd.DataFrame({'user_input': [user_input]})
    
    # Print the new DataFrame
    print(new_df_5)

# Create the GUI window
root = tk.Tk()

# Add a label widget and an entry widget to the GUI window
label = tk.Label(root, text='Enter your input: Safety/ Comfort')
label.pack()
entry = tk.Entry(root)
entry.pack()

# Add a button widget to the GUI window
button = tk.Button(root, text='Submit', command=get_input)
button.pack()

# Start the GUI event loop
root.mainloop()

user_input = 'safety'
print(high_safety(user_input))


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





  


