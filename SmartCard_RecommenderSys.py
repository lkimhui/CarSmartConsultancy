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
from sklearn.preprocessing import StandardScaler
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import datetime

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
print(df_nonull.info())  

#Building a sample recommender system for testing
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

#after initial recommender 1, it seems that an extensive data cleaning is require. Moving to data cleaning first

df_alllesstext= df

#after initial recommender 1, it seems that an extensive data cleaning is require. Moving to data cleaning first
#depreciation
df_alllesstext['depreciation'] = df_alllesstext['depreciation'].str.replace(',', '')
df_alllesstext['depreciation'] = df_alllesstext['depreciation'].str.replace('$', '')
df_alllesstext['depreciation'] = df_alllesstext['depreciation'].replace('N.A.', np.nan)
df_alllesstext['depreciation'] = df_alllesstext['depreciation'].replace('N.A', np.nan)
df_alllesstext['depreciation'] = df_alllesstext['depreciation'].astype(float)

#deregistration_value
df_alllesstext['deregistration_value'] = df_alllesstext['deregistration_value'].str.replace(',', '')
df_alllesstext['deregistration_value'] = df_alllesstext['deregistration_value'].str.replace('$', '')
df_alllesstext['deregistration_value'] = df_alllesstext['deregistration_value'].replace('N.A.', np.nan)
df_alllesstext['deregistration_value'] = df_alllesstext['deregistration_value'].astype(float)

#coe
df_alllesstext['coe'] = df_alllesstext['coe'].str.replace(',', '')
df_alllesstext['coe'] = df_alllesstext['coe'].str.replace('$', '')
df_alllesstext['coe'] = df_alllesstext['coe'].replace('N.A.', np.nan)
df_alllesstext['coe'] = df_alllesstext['coe'].astype(float)

#omv
df_alllesstext['omv'] = df_alllesstext['omv'].str.replace(',', '')
df_alllesstext['omv'] = df_alllesstext['omv'].str.replace('$', '')
df_alllesstext['omv'] = df_alllesstext['omv'].replace('N.A.', np.nan)
df_alllesstext['omv'] = df_alllesstext['omv'].astype(float)

#arf
df_alllesstext['arf'] = df_alllesstext['arf'].str.replace(',', '')
df_alllesstext['arf'] = df_alllesstext['arf'].str.replace('$', '')
df_alllesstext['arf'] = df_alllesstext['arf'].replace('N.A.', np.nan)
df_alllesstext = df_alllesstext[~df_alllesstext['arf'].str.contains('kW')]
df_alllesstext['arf'] = df_alllesstext['arf'].astype(float)

#roadtax
df_alllesstext['road_tax'] = df_alllesstext['road_tax'].str.replace(',', '')
df_alllesstext['road_tax'] = df_alllesstext['road_tax'].str.replace('$', '')
df_alllesstext['road_tax'] = df_alllesstext['road_tax'].str.replace('/yr', '')
df_alllesstext['road_tax'] = df_alllesstext['road_tax'].replace('N.A.', np.nan)
df_alllesstext['road_tax'] = df_alllesstext['road_tax'].astype(float)

#mileage
df_alllesstext['mileage'] = df_alllesstext['mileage'].str.replace(r'\(.*?\)', '', regex=True)
df_alllesstext['mileage'] = df_alllesstext['mileage'].str.replace(',', '')
df_alllesstext['mileage'] = df_alllesstext['mileage'].str.replace('$', '')
df_alllesstext['mileage'] = df_alllesstext['mileage'].str.replace('km', '')
df_alllesstext['mileage'] = df_alllesstext['mileage'].replace('N.A.', np.nan)
df_alllesstext['mileage'] = df_alllesstext['mileage'].astype(float)

#enginecap
df_alllesstext['engine_cap'] = df_alllesstext['engine_cap'].str.replace(',', '')
df_alllesstext['engine_cap'] = df_alllesstext['engine_cap'].str.replace('cc', '')
df_alllesstext['engine_cap'] = df_alllesstext['engine_cap'].replace('N.A.', np.nan)
df_alllesstext['engine_cap'] = df_alllesstext['engine_cap'].astype(float)

#curb_weight
df_alllesstext['curb_weight'] = df_alllesstext['curb_weight'].str.replace(',', '')
df_alllesstext['curb_weight'] = df_alllesstext['curb_weight'].str.replace('kg', '')
df_alllesstext['curb_weight'] = df_alllesstext['curb_weight'].replace('N.A.', np.nan)
df_alllesstext['curb_weight'] = df_alllesstext['engine_cap'].astype(float)

#power
df_alllesstext['power'] = df_alllesstext['power'].str.replace(r'\(.*?\)', '', regex=True)
df_alllesstext['power'] = df_alllesstext['power'].str.replace(',', '')
df_alllesstext['power'] = df_alllesstext['power'].str.replace('kW', '')
df_alllesstext['power'] = df_alllesstext['power'].replace('N.A.', np.nan)
df_alllesstext = df_alllesstext[~df_alllesstext['power'].str.contains('More than 6')]
df_alllesstext['power'] = df_alllesstext['power'].astype(float)

#registration_dates
# convert the 'registration_date' column to a datetime object
df_alllesstext['registration_date'] = pd.to_datetime(df_alllesstext['registration_date'], format='%d-%b-%y')
# calculate the difference between the registration date and today's date
today = datetime.datetime.now()
df_alllesstext['age_of_car'] = (today - df_alllesstext['registration_date']) / pd.Timedelta(days=365)
# show the updated DataFrame
print(df_alllesstext)

#manufacturered year
# calculate the age of the car based on the manufacturer year
df_alllesstext['manufactured_year'] = df_alllesstext['manufactured_year'].astype(float)
year = 2023
df_alllesstext['years_since_launch'] = year - df_alllesstext['manufactured_year']

#Transmission
# use get_dummies() to one-hot encode the 'trasmission' column
df_onehot = pd.get_dummies(df_alllesstext['trasmission'], prefix='trasmission')
# concatenate the one-hot encoded features with the original DataFrame
df_alllesstext = pd.concat([df_alllesstext, df_onehot], axis=1)
# drop the original 'transmission' column
df_alllesstext = df_alllesstext.drop('trasmission', axis=1)

#number_of_owner
# use get_dummies() to one-hot encode the 'number_of_owner' column
df_onehot = pd.get_dummies(df_alllesstext['number_of_owner'], prefix='number_of_owner')
# concatenate the one-hot encoded features with the original DataFrame
df_alllesstext = pd.concat([df_alllesstext, df_onehot], axis=1)
# drop the original 'transmission' column
df_alllesstext = df_alllesstext.drop('number_of_owner', axis=1)

#type
# use get_dummies() to one-hot encode the 'type' column
df_onehot = pd.get_dummies(df_alllesstext['type'], prefix='type')
# concatenate the one-hot encoded features with the original DataFrame
df_alllesstext = pd.concat([df_alllesstext, df_onehot], axis=1)
# drop the original 'transmission' column
df_alllesstext = df_alllesstext.drop('type', axis=1)

# show the updated DataFrame
print(df_alllesstext)

df_alllesstext.info()

df_alllesstext.dropna(inplace=True)

null_counts = df_alllesstext.isnull().sum()
print(null_counts)

#This is for data checking after cleaning
file_name = "checking.xlsx"
file_path = '/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data'
df_alllesstext.to_excel(file_path + file_name, index=False)

#Normalization #need to revisit the method
# columns to normalize
cols_to_normalize = ['price', 'depreciation', 'mileage', 'road_tax', 'deregistration_value', 'coe', 'engine_cap', 'curb_weight', 'omv', 'arf', 'power', 'age_of_car', 'years_since_launch']

# replace non-numeric values with NaN
df_alllesstext[cols_to_normalize] = df_alllesstext[cols_to_normalize].apply(pd.to_numeric, errors='coerce')

# create a StandardScaler object
scaler = StandardScaler()

# fit and transform the selected columns
df_alllesstext[cols_to_normalize] = scaler.fit_transform(df_alllesstext[cols_to_normalize])
#This is for data checking after cleaning
file_name = "checking.xlsx"
file_path = '/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data'
df_alllesstext.to_excel(file_path + file_name, index=False)

#Using features without text for the recommdner

df_touse = df_alllesstext.drop(['registration_date', 'car_features', 'car_accessories', 'descriptions'],axis =1)
df_touse.info()

#features to select




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




#Kwan Code
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





  


