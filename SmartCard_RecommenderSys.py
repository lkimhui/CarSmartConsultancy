# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

# Purpose: For cleaning data & build a recommender system
#### TO INCLUDE FEATURE ENGINEERING before finalizing RCS

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords #pip install nltk 
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

#Load the CSV file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
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
print(df_nonull.info())  #Note to kwan, I will use this part for the price recommender

features = df_nonull[['car_id','manufactured_year','mileage']]
print(features.describe())
print(features.info())
print(features.head())

#Data Cleaning 
features_cleaned = features.copy()

#Delimit Mileage to obtain only in km e.g., 124000
print(features['mileage'].dtype)

# Define a function to extract the integer mileage value
def extract_mileage(mileage_str):
    if mileage_str == 'N.A.':
        return None
    else:
        return int(mileage_str.replace(',', '').split()[0])

# Apply the function to the 'mileage' column
features_cleaned['mileage_int'] = features_cleaned['mileage'].apply(extract_mileage)
print(features_cleaned.head())

features_cleaned = features_cleaned[features_cleaned['mileage'] != 'N.A.']
features_cleaned.drop('mileage', axis=1, inplace=True)

top_50_cars_safety = features_cleaned.sort_values(['manufactured_year', 'mileage_int'], ascending=[False, True]).head(50)
print(top_50_cars_safety.head())
#11Apr: yay, able to get the top 50 cars with car_id based on manufactured_year and mileage.

#Remove punctuation from Price and Depreciation
#features_cleaned['price']= features_cleaned['price'].str.replace('[\$,]', '', regex=True)
#features_cleaned['price']= features_cleaned['price'].str.replace(',', '', regex=True)
#features_cleaned['depreciation']= features_cleaned['depreciation'].str.replace('[\$,]', '', regex=True)
#features_cleaned['depreciation']= features_cleaned['depreciation'].str.replace(',', '', regex=True)

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



#Building a few sample recommender system
df = df_nonull
print(df.head())
print(df.info())

df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].replace('N.A', np.nan)
df['price'] = df['price'].astype(float)

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

    
#2 Using all features less text from features, accessories and descriptions
df_allfeatureslesswords = df

df_allfeatureslesswords['price'] = df_allfeatureslesswords.groupby('model')['price'].transform(lambda x: x.fillna(x.mean()))
df_allfeatureslesswords = df_allfeatureslesswords.dropna(subset=['price'])

null_counts = df_allfeatureslesswords.isnull().sum()
print(null_counts)

df_allfeatureslesswords = df_allfeatureslesswords.drop(['car_features', 'car_accessories', 'descriptions'], axis = 1)
df_allfeatureslesswords['depreciation', 'road_tax'] = df_allfeatureslesswords['depreciation', 'road_tax'].str.replace(',', '')
df_allfeatureslesswords['price'] = df_allfeatureslesswords['price'].str.replace('$', '')
columns_to_float = ['depreciation', 'mileage', 'road_tax', 'deregistration_value', 'coe', 'engine_cap', 'curb_weight', 'trasmission', 'omv', 'arf', 'power', 'number_of_owner']
df_allfeatureslesswords[columns_to_float] = df_allfeatureslesswords[columns_to_float].astype(float)
df_allfeatureslesswords.info()

#3 Using just the words description

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
model_name = 'Volvo V60 T5 R-Design'
top_similar_models = cos_sim_df[[model_name]].sort_values(by=model_name, ascending=False)[1:6]

print(top_similar_models)

print(df.head())








  


