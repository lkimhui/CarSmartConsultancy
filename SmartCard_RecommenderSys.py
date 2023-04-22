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
from nltk.corpus import stopwords #pip install nltk 
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import datetime
import tkinter as tk

## Adding latest cleaned and rolled up data here ## 

#Load the CSV file

#preview the csv file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
#df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
print(df.describe())
print(df.info())
print(df.head())

##5 Recommender in total ##

#1.) revisting the 1st model using just price on the clean dataset

df_1 = df

df_1 = df[['car_id', 'model', 'price']]
df_1['price'] = df_1.groupby('model')['price'].transform(lambda x: x.fillna(x.mean()))
df_1 = df_1.dropna(subset=['price'])
null_counts = df_1.isnull().sum()
print(null_counts)
print(df_1.head())
print(df_1.info())

# Define the Reader object to read the data
reader = Reader(rating_scale=(df_1['price'].min(), df_1['price'].max()))

# Load the data from the dataframe into the Surprise Dataset object
data = Dataset.load_from_df(df_1[['model', 'car_id', 'price']], reader=reader)

# Define the SVD algorithm and fit the training data
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Define a function to get recommended products
def get_recommendations(price):
    # Create a list of items that the user has not yet rated
    model_list = df_1[df_1['price'] <= price]['model'].unique().tolist()
    all_models = df_1['model'].unique().tolist()
    unrated_models = list(set(all_models) - set(model_list))
    unrated_ids = df_1[df_1['model'].isin(unrated_models)]['car_id'].unique().tolist()

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
        model = df_1[df_1['car_id'] == car_id]['model'].values[0]
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


#The output means that the mean RMSE (Root Mean Squared Error) of the model is 185201.348 vs 2090304.553 previously and the mean MAE (Mean Absolute Error) is 104722.119 vs 1791902.207 previously.
#This is a stark improvement from the model prior to cleaning in our first try. However, we need to reference further to other model and utilize the other features to compare 
#We will explore 4 more methods here

#2.) Similarity based on product text descritpion columns

#Mining the text data
#1 Using just the words description to build a recommdner

df_2 = df
df_2.info()
#Converting features to string
df_2['features'] = df_2['features'].astype(str)
df_2['accessories'] = df_2['accessories'].astype(str)
df_2['descriptions'] = df_2['descriptions'].astype(str)


# define the columns to clean
cols_to_clean = ['features', 'accessories', 'descriptions']

# convert the specified columns to lowercase
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.lower())

# remove numbers and unwanted characters using regular expressions
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.replace(r'[^a-z\s]', '', regex=True))

# tokenize the specified columns into words
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.split())

# remove stop words using nltk library
stop_words = set(stopwords.words('english'))
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: [word for word in y if not pd.isnull(word) and word not in stop_words]))

# lemmatize the words using nltk library
lemmatizer = WordNetLemmatizer()
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: [lemmatizer.lemmatize(word) for word in y]))

# join the words back into a single string
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: ' '.join(y)))

# fill NaN values with empty string
#df[cols_to_clean] = df[cols_to_clean].replace(np.nan, '', regex=True)

# vectorize the cleaned up columns using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_2[cols_to_clean].apply(lambda x: ' '.join(x), axis=1))

# print the resulting feature names and document-term matrix
print(vectorizer.get_feature_names())
print(X.toarray())


# create a dataframe from the resulting document-term matrix
tfidf_df = pd.DataFrame(X.toarray(),
                        columns=vectorizer.get_feature_names())
tfidf_df.index = df_2['model']

print(tfidf_df)

# calculate cosine similarity between each pair of rows in the tfidf_df dataframe
cos_sim = cosine_similarity(tfidf_df)

# create a new dataframe with the cosine similarity scores
cos_sim_df = pd.DataFrame(cos_sim, columns=tfidf_df.index, index=tfidf_df.index)

# display the top 10 most similar models to a given model (in this example, model 'Honda Vezel')
model_name = 'Volvo V60 T5 R-Design'
top_similar_models = cos_sim_df[[model_name]].sort_values(by=model_name, ascending=False)[1:6]

print(top_similar_models)

print(df_2.head())

#3.) base on all non text features

df_3 = df
df_3 = df.drop(columns = ['features', 'accessories', 'descriptions'])
df_3.info()

# Select the columns containing the features that will be used for recommendation
features = df_3.drop(['car_id', 'model'], axis=1)

# Compute the pairwise cosine similarities between the cars
similarities = cosine_similarity(features)

# Define a function to recommend similar cars based on a given car model name
def recommend_car(model_name):
    # Get the row index corresponding to the given car model
    model_index = df_3.index[df_3['model'] == model_name][0]

    # Get the pairwise cosine similarities between the given car and all other cars
    car_similarities = similarities[model_index]

    # Get the indices of the top 5 most similar cars (excluding the given car itself)
    top_indices = car_similarities.argsort()[:-6:-1][1:]

    # Return the top 5 most similar cars
    return df_3.iloc[top_indices][['car_id', 'model']]

# Example usage: recommend 5 cars similar to the car model "Toyota Camry"
print(recommend_car("Volvo V60 T5 R-Design"))

#4.) base on combine feature, text and non text (This is for user who dunno what they want)
df_4 = df
df_4.info()

# Define the stemmer and stop words list
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# Define a function to tokenize, remove stop words, and stem the text
def process_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words and stem the tokens
    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    # Return the processed tokens as a string
    return " ".join(processed_tokens)
    
# Apply the function to the columns containing textual data
df_4["processed_features"] = df_4["features"].apply(process_text)
df_4["processed_accessories"] = df_4["accessories"].apply(process_text)
df_4["processed_descriptions"] = df_4["descriptions"].apply(process_text)

# Tokenize the text data in the "features", "accessories", and "descriptions" columns
df_4['features_tokens'] = df_4['features'].apply(nltk.word_tokenize)
df_4['accessories_tokens'] = df_4['accessories'].apply(nltk.word_tokenize)
df_4['descriptions_tokens'] = df_4['descriptions'].apply(nltk.word_tokenize)

# Define a function to count the occurrence of each word in a list of tokens
def count_tokens(tokens):
    return Counter(tokens)

# Count the occurrence of each word in the "features_tokens", "accessories_tokens", and "descriptions_tokens" columns
df_4['features_word_counts'] = df_4['features_tokens'].apply(count_tokens)
df_4['accessories_word_counts'] = df_4['accessories_tokens'].apply(count_tokens)
df_4['descriptions_word_counts'] = df_4['descriptions_tokens'].apply(count_tokens)

df_4 = df_4.drop(["features", "accessories", "descriptions"], axis=1)
df_4.info()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_4.drop('price', axis=1), df_4['price'], test_size=0.2, random_state=42)

#In a hybrid recommender system, we use both the user's preferences and the item features to make recommendations. The content-based approach uses item features to find similar items, while collaborative filtering uses user-item interactions to find similar users or items

# Define the columns to be used for similarity
columns_to_use = ['processed_features', 'processed_accessories', 'processed_descriptions', 'model', 'transmission_Auto', 'transmission_Manual', 'types_Hatchback', 'types_Luxury Sedan', 'types_MPV', 'types_Mid-Sized Sedan', 'types_SUV', 'types_Sports Car', 'types_Stationwagon', 'types_Truck', 'types_Van', 'power', 'age_of_car', 'price']

# Combine the columns into a single string
df_4['combined_features'] = df_4[columns_to_use].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Define the count vectorizer and fit it to the combined features
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df_4['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def get_top_5_car_models(model_name, cosine_sim=cosine_sim, df_4=df_4):
    # Get the indices of the cars with the given model name
    car_indices = df[df['model'] == model_name].index
    
    # If no cars are found with the given model name, return an empty dataframe
    if len(car_indices) == 0:
        return pd.DataFrame()
    
    # Get the pairwise similarity scores of the cars with the given model name
    sim_scores = list(enumerate(cosine_sim[car_indices[0]]))
    
    # Sort the cars based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top 5 similar cars
    top_car_indices = [i[0] for i in sim_scores[1:6]]
    
    # Get the model names of the top 5 similar cars
    top_car_models = df.iloc[top_car_indices]['model'].values.tolist()
    
    # Return the top 5 similar car models
    return top_car_models

get_top_5_car_models('Volvo V60 T5 R-Design')

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

#6.) Showcsing the top 10 selling model in the past periods
#Top selling models
df_6 = pd.read_csv("/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/TableauData.csv")
df_6.info()
sold_counts = df_6[df_6['status'] == 'SOLD'].groupby('model')['status'].count().reset_index(name='sold_count').sort_values(by='sold_count', ascending=False)
print(sold_counts.head(10))


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





  


