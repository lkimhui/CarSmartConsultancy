# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

##Start##

##Recommender Systems##

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from math import sqrt
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import datetime
import tkinter as tk
from PIL import Image, ImageTk #need to pip install Pillow
import matplotlib.pyplot as plt

## Adding latest cleaned and rolled up data here ## 

#Load the CSV file

#preview the csv file
#df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/datarollup_latest.csv')
print(df.describe())
print(df.info())
print(df.head())

#visual
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
table_data = [    ['Column Name', 'Data Type'],
    *[[col, df[col].dtype] for col in df.columns]
]

table = ax.table(cellText=table_data, loc='center')
table.set_fontsize(14)
table.scale(1, 2)
ax.axis('off')
plt.show()

##6 Recommender in total ##
## To show case 3,4,5,6 ##

#1.) revisting the 1st model using just price on the clean dataset

# Load the dataset into a pandas dataframe
df_1 = df.copy()

df_1 = df[['car_id', 'model', 'price']]
df_1['price'] = df_1.groupby('model')['price'].transform(lambda x: x.fillna(x.mean()))
df_1 = df_1.dropna(subset=['price'])
null_counts = df_1.isnull().sum()
print(null_counts)
print(df_1.head())
print(df_1.info())

# Define the Reader object to read the data
reader = Reader(rating_scale=(df['price'].min(), df['price'].max()))

# Load the data from the dataframe into the Surprise Dataset object
data = Dataset.load_from_df(df[['model', 'car_id', 'price']], reader=reader)

# Define the SVD algorithm and fit the training data
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Define a function to get recommended products
def get_recommendations(budget):
    # Create a list of items that the user can afford
    affordable_models = df[df['price'] <= budget]['model'].unique().tolist()
    
    # Filter the data to include only the affordable models
    df_affordable = df[df['model'].isin(affordable_models)]
    
    # Group the data by model and fill missing prices with the mean price for the model
    df_grouped = df_affordable.groupby('model').apply(lambda x: x.fillna(x.mean()))
    
    # Drop any rows with missing data
    df_filtered = df_grouped.dropna(subset=['price'])
    
    # Load the filtered data into the Surprise Dataset object
    data_filtered = Dataset.load_from_df(df_filtered[['model', 'car_id', 'price']], reader=reader)
    
    # Build the full training set and fit the algorithm
    trainset_filtered = data_filtered.build_full_trainset()
    algo.fit(trainset_filtered)
    
    # Get the unrated car ids for the affordable models
    unrated_ids = df_filtered[~df_filtered['car_id'].isin(trainset_filtered.all_items())]['car_id'].unique().tolist()
    
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
    for prediction in top_predictions:
        if len(prediction) == 2:
            car_id, rating = prediction
            model = df_filtered[df_filtered['car_id'] == car_id]['model'].values[0]
            recommendations.append((model, rating))
    
    return recommendations

# Example usage
budget = 50000
recommendations = get_recommendations(budget)
print(f"Recommended models within budget of {budget}:")
for rec in recommendations:
    print(f"Model: {rec[0]}, Predicted Rating: {rec[1]}")  #For 50,000, model recommended BMW 2 or 3 series. 

budget = 35000
recommendations = get_recommendations(budget)
print(f"Recommended models within budget of {budget}:")
for rec in recommendations:
    print(f"Model: {rec[0]}, Predicted Rating: {rec[1]}")
    
budget = 15000
recommendations = get_recommendations(budget)
print(f"Recommended models within budget of {budget}:")
for rec in recommendations:
    print(f"Model: {rec[0]}, Predicted Rating: {rec[1]}")  
    
budget = 90000
recommendations = get_recommendations(budget)
print(f"Recommended models within budget of {budget}:")
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
#This is a stark improvement from the model prior to cleaning in our first try. 
#Also, recommender 1 is solely base on price and while it showcases some recommender techniques, the business logic is no difference from a filtering system
#However, we need to reference further to other model and utilize the other features to compare 
#We will explore 4 more methods here

 #2.) Similarity based on product text descritpion columns

#Mining the text data
#1 Using just the words description to build a recommdner

df_2 = df[['model','features','accessories','descriptions']].copy()
df_2.loc[:, 'features'] = df_2['features'].astype(str)
df_2.loc[:, 'accessories'] = df_2['accessories'].astype(str)
df_2.loc[:, 'descriptions'] = df_2['descriptions'].astype(str)
print(df_2.dtypes)
print(df_2.head())

# convert the specified columns to lowercase
cols_to_clean = ['features', 'accessories', 'descriptions']
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.lower())

# define function to tokenize, remove stop words, and lemmatize
def clean_text(text):
    # tokenize text into words
    words = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# apply the clean_text function to the specified columns
df_2[cols_to_clean] = df_2[cols_to_clean].applymap(clean_text)

# remove numbers and unwanted characters using regular expressions
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.replace(r'[^a-z\s]', '', regex=True))

#tokenize the specified columns into words, remove stop words, lemmatize the words, and join them back into a single string:
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.str.split())
stop_words = set(stopwords.words('english'))
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: [word for word in y if not pd.isnull(word) and word not in stop_words]))
lemmatizer = WordNetLemmatizer()
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: [lemmatizer.lemmatize(word) for word in y]))
df_2[cols_to_clean] = df_2[cols_to_clean].apply(lambda x: x.apply(lambda y: ' '.join(y)))

# print the cleaned up data
print(df_2.head())

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

# display the top 5 most similar models to a given model (in this example, model 'Volvo V60 T5 R-Design')
model_name = 'Volvo V60 T5 R-Design'
top_similar_models = cos_sim_df[[model_name]].sort_values(by=model_name, ascending=False)[1:6]

print(top_similar_models)

model_name = 'Toyota Alphard Hybrid 2.5A X'
top_similar_models = cos_sim_df[[model_name]].sort_values(by=model_name, ascending=False)[1:6]

print(top_similar_models)

#3.) base on all non text features

df_3 = df.copy()
df_3 = df.drop(columns = ['features', 'accessories', 'descriptions'])
df_3.info()

# Select the columns containing the features that will be used for recommendation
features = df_3.drop(['car_id', 'model'], axis=1)

# Compute the pairwise cosine similarities between the cars
similarities = cosine_similarity(features)

def recommend_car(model_name):
    # Get the row index corresponding to the given car model
    model_index = df_3.index[df_3['model'] == model_name][0]

    # Get the pairwise cosine similarities between the given car and all other cars
    car_similarities = similarities[model_index]

    # Get the indices of the top 5 most similar cars (excluding the given car itself)
    top_indices = car_similarities.argsort()[:-7:-1][1:]

    # Get the cosine similarity scores of the top 5 most similar cars
    top_similarities = car_similarities[top_indices]

    # Return the top 5 most similar cars and their cosine similarity scores
    return df_3.iloc[top_indices][['car_id', 'model']], top_similarities

# Example usage: recommend 5 cars similar to the car model "Volvo V60 T5 R-Design"
similar_cars, similarities = recommend_car("Volvo V60 T5 R-Design")
print(similar_cars)
print(similarities)

similar_cars, similarities = recommend_car("Toyota Alphard Hybrid 2.5A X")
print(similar_cars)
print(similarities)

similar_cars, similarities = recommend_car("Lamborghini Aventador LP700-4 ")
print(similar_cars)
print(similarities)

#Recommender 3 without type features, number of owners etc..

df_3 = df.copy()
df_3 = df.drop(columns = ['features', 'accessories', 'descriptions'])
df_3.info()

# Select the columns containing the features that will be used for recommendation
features = df_3.drop(['car_id', 'model', 'number_of_owner_More than 6','types_Hatchback', 'types_Luxury Sedan', 'types_MPV', 'types_Mid-Sized Sedan', 'types_SUV', 'types_Sports Car', 'types_Stationwagon', 'types_Truck', 'types_Van'], axis=1)

# Compute the pairwise cosine similarities between the cars
similarities = cosine_similarity(features)

def recommend_car(model_name):
    # Get the row index corresponding to the given car model
    model_index = df_3.index[df_3['model'] == model_name][0]

    # Get the pairwise cosine similarities between the given car and all other cars
    car_similarities = similarities[model_index]

    # Get the indices of the top 5 most similar cars (excluding the given car itself)
    top_indices = car_similarities.argsort()[:-7:-1][1:]

    # Get the cosine similarity scores of the top 5 most similar cars
    top_similarities = car_similarities[top_indices]

    # Return the top 5 most similar cars and their cosine similarity scores
    return df_3.iloc[top_indices][['car_id', 'model']], top_similarities

# Example usage: recommend 5 cars similar to the car model "Volvo V60 T5 R-Design"
similar_cars, similarities = recommend_car("Volvo V60 T5 R-Design")
print(similar_cars)
print(similarities)

similar_cars, similarities = recommend_car("Toyota Alphard Hybrid 2.5A X")
print(similar_cars)
print(similarities)

similar_cars, similarities = recommend_car("Lamborghini Aventador LP700-4 ")
print(similar_cars)
print(similarities)


#This recommended looks good with 99% similarities. 

#4.) base on combine feature, text and non text (This is for user who dunno what they want)
df_4 = df.copy()
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

#In a hybrid recommender system, we use both the user's preferences and the item features to make recommendations. The content-based approach uses item features to find similar items, while collaborative filtering uses user-item interactions to find similar users or items

#Still got the and only etc..
#Feature selection code


# Define the columns to be used for similarity
#columns_to_use = ['processed_features', 'processed_accessories', 'processed_descriptions', 'model', 'transmission_Auto', 'transmission_Manual', 'types_Hatchback', 'types_Luxury Sedan', 'types_MPV', 'types_Mid-Sized Sedan', 'types_SUV', 'types_Sports Car', 'types_Stationwagon', 'types_Truck', 'types_Van', 'power', 'age_of_car', 'price']
columns_to_use = ['processed_features', 'processed_accessories', 'model', 'transmission_Auto', 'transmission_Manual', 'types_Hatchback', 'types_Luxury Sedan', 'types_MPV', 'types_Mid-Sized Sedan', 'types_SUV', 'types_Sports Car', 'types_Stationwagon', 'types_Truck', 'types_Van', 'power', 'age_of_car', 'price']

# Combine the columns into a single string
df_4['combined_features'] = df_4[columns_to_use].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Define the count vectorizer and fit it to the combined features
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df_4['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def get_top_5_car_models(model_name, cosine_sim=cosine_sim, df_4=df_4):
    # Get the indices of the cars with the given model name
    car_indices = df_4[df_4['model'] == model_name].index
    
    # If no cars are found with the given model name, return an empty dataframe
    if len(car_indices) == 0:
        return pd.DataFrame()
    
    # Get the pairwise similarity scores of the cars with the given model name
    sim_scores = list(enumerate(cosine_sim[car_indices[0]]))
    
    # Sort the cars based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices and cosine similarity scores of the top 5 similar cars
    top_car_indices = [i[0] for i in sim_scores[1:6]]
    top_car_scores = [i[1] for i in sim_scores[1:6]]
    
    # Get the model names of the top 5 similar cars
    top_car_models = df_4.iloc[top_car_indices]['model'].values.tolist()
    
    # Print the top 5 similar car models with their respective cosine similarity scores
    for i, model in enumerate(top_car_models):
        print(f"{i+1}. {model} (Cosine Similarity Score: {top_car_scores[i]:.2f})")
    
    # Return the top 5 similar car models
    return top_car_models

get_top_5_car_models("Volvo V60 T5 R-Design")

get_top_5_car_models("Toyota Alphard Hybrid 2.5A X")

similar_cars, similarities = recommend_car("Lamborghini Aventador LP700-4 ")
print(similar_cars)
print(similarities)

# 5.) based on user input, price, car type and age (This is for user who roughly know what they want)

# Higher Safety Rating 
# Reasoning: 
    # Car Type (SUV,  Pickup Truck) has Better safety profile; generally safer in collisions than smaller cars.
    # Newer cars usally have better safety features and technology


# Higher Comfort Rating 
# Reasoning: 
    # Transmission (Automatic)
    # Car Type (Luxury) may give more advanced comfort features.
    # Newer launched cars usally have  newer cars may have better suspension, quieter cabins, and more advanced technology

df_5 = df.copy()
print(df_5.head())

# Attempt getting user input using Tkinter (a Python GUI toolkit)

# define safety and comfort keywords
safety_keywords = ['airbag', 'anti-lock', 'blind spot', 'sensor', 'camera', 'warning', 'safety', 'braking', 'beam', 'smart']
comfort_keywords = ['luxurious', 'comfort', 'climate', 'infotainment system', 'knockdown','smart']

# tokenize and check for safety/comfort keywords
df_5[['features', 'accessories','descriptions']] = df_5[['features', 'accessories','descriptions']].applymap(lambda x: ' '.join(x) if isinstance(x, list) else x)

print(df_5['features'].dtype)
print(df_5['accessories'].dtype)

df_5['text'] = df_5['features'] + df_5['accessories'] + df_5['descriptions']
df_5['tokens'] = df_5['text'].str.lower().str.split()
df_5['safety_score'] = df_5['tokens'].apply(lambda x: sum(1 for w in x if w in safety_keywords))
df_5['comfort_score'] = df_5['tokens'].apply(lambda x: sum(1 for w in x if w in comfort_keywords))


# function to handle user input

def selection(user_input):
    if user_input == 'Safety':
        print("Safety is preferred and here are the top cars which have a high safety rating")
        suv_pickup_cars = df_5[(df_5['types_SUV'] == 1) | (df_5['types_Truck'] == 1)]
        top_50_cars_safety = suv_pickup_cars.sort_values(['age_of_car', 'safety_score'], ascending=[True, False]).head(50)
        #output = print(top_50_cars_safety.head())
        output = print(top_50_cars_safety[['model', 'safety_score']].head())
 
        # concatenate safety keywords in top_50_cars_safety
        safety_words = ' '.join([word for word in top_50_cars_safety['tokens'].explode() if word in safety_keywords])

        # create wordcloud
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate(safety_words)

        # plot the wordcloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
                
    elif user_input == 'Comfort':
        print("Comfort is preferred and here are the top cars which have a high comfort rating")
        luxury_auto_cars = df_5[(df_5['types_Luxury Sedan'] == 1) & (df_5['transmission_Auto'] == 1)]
        top_50_cars_comfort = luxury_auto_cars.sort_values(['years_since_launch', 'comfort_score'], ascending=[True, False]).head(50)
        #output = print(top_50_cars_comfort.head())
        output = print(top_50_cars_comfort[['model','comfort_score']].head())   
        
        # concatenate safety keywords in top_50_cars_comfort
        comfort_words = ' '.join([word for word in top_50_cars_comfort['tokens'].explode() if word in comfort_keywords])

        # create wordcloud
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate(comfort_words)

        # plot the wordcloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        
    else:
        print("Invalid input. Please select only 1 option.")

selection('Comfort') #test
file_name = f"rcs5_test.csv"
#file_path = '/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
file_path = '/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
df_5.to_csv(file_path + file_name, index=False)


# Create the GUI window

# create tkinter window

root = tk.Tk()
root.geometry("500x450")
root.configure(bg='white')
root.title("User selection of Car Preference")

label = tk.Label(root, text="Select your preference:")
label.pack()

# Create a PhotoImage object from a file
image_file = Image.open('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/CSC_Logo.png')
image = ImageTk.PhotoImage(image_file)

# Create a Label widget to display the image
image_label = tk.Label(root, image=image)
image_label.pack()
image_label.image = image  # keep a reference to the image

# Create selection buttons
var = tk.StringVar()

radio_button1 = tk.Radiobutton(root, text="Safety", variable=var, value='Safety', font=("Cambria", 14))
radio_button1.pack()

radio_button2 = tk.Radiobutton(root, text="Comfort", variable=var, value='Comfort', font=("Cambria", 14))
radio_button2.pack()

submit_button = tk.Button(root, text="Submit", command=lambda: [selection(var.get()), root.destroy()], font=("Cambria", 14), fg="white", bg="black")
submit_button.pack()

# run the window loop
root.mainloop()


#6.) Showcasing the top 10 selling model in the past periods
#Popularity based recommender
#Top selling models

# Create the DataFrame and perform the aggregation as before
df_6 = pd.read_csv("/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/TableauData.csv")

df_6 = pd.read_csv("/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/TableauData.csv")
df_6.info()

sold_counts = df_6[df_6['status'] == 'SOLD'].groupby('model')['status'].count().reset_index(name='sold_count').sort_values(by='sold_count', ascending=False)


#Creating a visual
# Reverse the order of the DataFrame
sold_counts = sold_counts.iloc[::-1]

# Set up the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create the horizontal bar chart with turquoise color
bars = ax.barh(y=sold_counts['model'].tail(10), width=sold_counts['sold_count'].tail(10), color='turquoise')

# Set the axis labels and title
ax.set_xlabel('Number of Sales')
ax.set_ylabel('Car Model')
ax.set_title('Top 10 Best-Selling Car Models')

# Sort the bars in ascending order
bars = sorted(bars, key=lambda x: x.get_width())

# Add data labels to the bars
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
            str(int(bar.get_width())), ha='left', va='center')

# Show the plot
plt.show()


#7.) Features only, using status as the implicit feedback

#df_7a = pd.read_csv("/Users/ivanong/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/TableauData.csv")
df_7a = pd.read_csv("/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/Data/cleaned_data/TableauData.csv")

#Add unique ID(new column)
#Generate a sequence of integers for the unique IDs
ids = range(1, len(df_7a)+1)
df_7a['car_id']= ids
df_7a['car_id_URL'] = 'https://www.sgcarmart.com/used_cars/info.php?ID=' + df_7a['car_id'].astype(str)

print(df_7a.head())

df_7 = df_7a.copy()

df_7['status'].unique()

#Converting SOLD to 1 and AVAIL to 0
# create a dictionary to map the status values to 1 or 0
status_dict = {'SOLD': 1, 'AVAIL': 0}

# create a new column 'sold_indicator' based on 'status' column
df_7['sold_indicator'] = df_7['status'].map(status_dict)
df_7.info()

#Feature selection

# Drop columns dates
df_7.drop(['registration_date', 'manufactured_year'], axis=1, inplace=True)

#Using K best method
# Encode categorical columns
#cat_cols = ['model', 'transmission', 'types', 'status', 'maker']
#for col in cat_cols:
 #   le = LabelEncoder()
  #  df_7[col] = le.fit_transform(df_7[col])
    
# Define features and target
#X = df_7.drop(['sold_indicator'], axis=1)
#y = df_7['sold_indicator']

# Compute correlation between features and target
#corr_with_target = abs(X.corrwith(y))

# Compute mutual information between features and target
#mi = mutual_info_classif(X, y, random_state=0)

# Combine correlation and mutual information scores
#scores = corr_with_target + mi

# Select top k features
#k = 10
#selector = SelectKBest(k=k)
#selector.fit(X, y)
#selected_features = X.columns[selector.get_support()]

# Print the selected features
#print(selected_features)

#We have decided to use Correlation Selection as K best is comparatively more heristic in nature 
#You may try k Best method with the code above

#Using Correlation Selection
# Compute the correlation matrix
corr_matrix = df_7.corr()

# Get the absolute values of the correlations
corr_abs = corr_matrix.abs()

# Set the threshold for selecting correlated features
corr_threshold = 0.8

# Select upper triangle of correlation matrix
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(np.bool))

# Get the features to drop
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

# Drop the correlated features
df_7 = df_7.drop(to_drop, axis=1)
df_7.info()

#Recommender 7 based on only selected features

# Convert categorical columns to numerical using LabelEncoder
cat_cols = ['transmission', 'types', 'status', 'maker']
for col in cat_cols:
    le = LabelEncoder()
    df_7[col] = le.fit_transform(df_7[col])

# Get the target model
#target_model = 'Toyota Alphard Hybrid 2.5A X'
target_model = 'Volvo V60 T5 R-Design'
target_index = df_7[df_7['model'] == target_model].index[0]

# Convert the dataframe to a list of strings
model_list = df_7.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

# Convert the list to a sparse matrix of token counts
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(model_list)

# Calculate the cosine similarity between the target model and all other models
cosine_sim = cosine_similarity(count_matrix[target_index], count_matrix)

# Get the indices of the most similar models
sim_indices = cosine_sim.argsort()[0][-10:-1]

# Print the recommended models
print("Recommended Models:")
for i in sim_indices:
    if df_7.loc[i, 'sold_indicator'] == 0:
        print("- Model:", df_7['model'][i], "(Cosine Similarity:", cosine_sim[0][i], ")", df_7['car_id_URL'][i])


#Notes
#The first part of the code encodes the categorical features of the dataset (i.e., 'transmission', 'types', 'status', and 'maker') using a LabelEncoder object. This is necessary because the cosine similarity calculation later on requires numerical values as inputs.
#Next, the code sets a target model (target_model) and finds its index in the dataset (target_index). It then converts the entire dataset into a list of strings using the apply() method, which concatenates all the columns of each row into a single string.
#Afterwards, the list of strings is transformed into a sparse matrix of token counts using the CountVectorizer object, which tokenizes each string into individual words and counts their occurrences in the entire list.
#Finally, the cosine similarity between the target model and all other models in the dataset is calculated using the cosine_similarity function. The top 5 most similar models are identified based on their cosine similarity scores, and their names are printed as recommendations.


#Recommender 8, based on user preference on just Brand, Price and Age
df_8 = df.copy()
df_8.info()

# Define a function to get the indices of the top 5 recommendations
def recommend(index, df_8, column):
    # Compute the cosine similarity matrix
    sim_matrix = cosine_similarity(df_8[column])

    # Get the pair-wise similarity scores of the cars with the input car
    sim_scores = list(enumerate(sim_matrix[index]))

    # Sort the cars based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 similar cars
    top_indices = [i[0] for i in sim_scores[1:6]]

    # Return the top 5 similar cars
    return df_8.iloc[top_indices]

# Get the user's preferred brand, price and age of car
preferred_brand = "Honda"
preferred_price = 80000
preferred_age = 6

# Filter the dataset based on the user's preferences
filtered_df = df_8[(df_8['model'].str.contains(preferred_brand)) & (df_8['price'] <= preferred_price) & (df_8['age_of_car'] <= preferred_age)]

if not filtered_df.empty:
    # Reset the index of the filtered dataframe
    filtered_df = filtered_df.reset_index(drop=True)

    # Get the top 5 recommended cars
    recommended_cars = recommend(0, filtered_df, ['price', 'age_of_car'])

    # Display the recommended cars
    print("Recommended Cars:")
    print(recommended_cars)
else:
    print("No cars match the user's preferences.")

# Get the user's preferred brand, price and age of car
preferred_brand = "Toyota"
preferred_price = 50000
preferred_age = 5

# Filter the dataset based on the user's preferences
filtered_df = df_8[(df_8['model'].str.contains(preferred_brand)) & (df_8['price'] <= preferred_price) & (df_8['age_of_car'] <= preferred_age)]

if not filtered_df.empty:
    # Reset the index of the filtered dataframe
    filtered_df = filtered_df.reset_index(drop=True)

    # Get the top 5 recommended cars
    recommended_cars = recommend(0, filtered_df, ['price', 'age_of_car'])

    # Display the recommended cars
    print("Recommended Cars:")
    print(recommended_cars)
else:
    print("No cars match the user's preferences.")  ##If no product matches user preference
    
#In the case, we can configure the recommender system such that it recommend the next best product based on the price, brand or age of car
#lets do a test based on just price and brand accordingly

#Price
# Define a function to get the indices of the top 5 recommendations
def recommend(index, df_8, column):
    # Compute the cosine similarity matrix
    sim_matrix = cosine_similarity(df_8[column])

    # Get the pair-wise similarity scores of the cars with the input car
    sim_scores = list(enumerate(sim_matrix[index]))

    # Sort the cars based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 similar cars
    top_indices = [i[0] for i in sim_scores[1:6]]

    # Return the top 5 similar cars
    return df_8.iloc[top_indices]

# Get the user's preferred brand, price and age of car
preferred_brand = "Toyota"
preferred_price = 50000
preferred_age = 5

# Filter the dataset based on the user's preferences
filtered_df = df_8[(df_8['model'].str.contains(preferred_brand)) & (df_8['price'] <= preferred_price) & (df_8['age_of_car'] <= preferred_age)]

# Check if any cars match the user's preferences
if len(filtered_df) == 0:
    # If no cars match, recommend the next closest based on price
    filtered_df = df_8[(df_8['price'] > preferred_price) & (df_8['age_of_car'] <= preferred_age)]
    filtered_df = filtered_df.sort_values(['price'], ascending=[True])
    filtered_df = filtered_df.reset_index(drop=True)
    recommended_cars = filtered_df.iloc[0:5]
    print("No cars match your preferences. Here are the next closest options based on price:")
    print(recommended_cars)
else:
    # If cars match, get the top 5 recommended cars
    filtered_df = filtered_df.reset_index(drop=True)
    recommended_cars = recommend(0, filtered_df, ['price', 'age_of_car'])
    print("Recommended Cars:")
    print(recommended_cars)

#Brand
# Define a function to get the indices of the top 5 recommendations
def recommend(index, df_8, column):
    # Compute the cosine similarity matrix
    sim_matrix = cosine_similarity(df_8[column])

    # Get the pair-wise similarity scores of the cars with the input car
    sim_scores = list(enumerate(sim_matrix[index]))

    # Sort the cars based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 similar cars
    top_indices = [i[0] for i in sim_scores[1:6]]

    # Return the top 5 similar cars
    return df_8.iloc[top_indices]

# Get the user's preferred brand, price and age of car
preferred_brand = "Toyota"
preferred_price = 50000
preferred_age = 5

# Filter the dataset based on the user's preferences
filtered_df = df_8[(df_8['model'].str.contains(preferred_brand)) & (df_8['price'] <= preferred_price) & (df_8['age_of_car'] <= preferred_age)]

# Check if any cars match the user's preferences
if len(filtered_df) == 0:
    # If no cars match, recommend the next best based on brand
    filtered_df = df_8[df_8['model'].str.contains(preferred_brand)]
    filtered_df = filtered_df.sort_values('price', ascending=True)
    filtered_df = filtered_df.reset_index(drop=True)
    recommended_cars = filtered_df.iloc[:5]
    print("No cars match your preferences. Here are the next best options based on brand:")
    print(recommended_cars)
else:
    # If cars match, get the top 5 recommended cars
    filtered_df = filtered_df.reset_index(drop=True)
    recommended_cars = recommend(0, filtered_df, ['price', 'age_of_car'])
    print("Recommended Cars:")
    print(recommended_cars)


# 5A.) based on user input, price, car type and age (This is for user who roughly know what they want)

df_5A = df.copy()
print(df_5A.head())

# Attempt getting user input using Tkinter (a Python GUI toolkit)

# define safety and comfort keywords
safety_keywords = ['airbag', 'anti-lock', 'blind spot', 'sensor', 'camera', 'warning', 'safety', 'braking', 'beam', 'smart']
comfort_keywords = ['luxurious', 'comfort', 'climate', 'infotainment system', 'knockdown', 'smart']

# tokenize and check for safety/comfort keywords
df_5A[['features', 'accessories', 'descriptions']] = df_5A[['features', 'accessories', 'descriptions']].applymap(lambda x: ' '.join(x) if isinstance(x, list) else x)

df_5A['text'] = df_5A['features'] + df_5A['accessories'] + df_5A['descriptions']
df_5A['tokens'] = df_5A['text'].str.lower().str.split()
df_5A['safety_score'] = df_5A['tokens'].apply(lambda x: sum(1 for w in x if w in safety_keywords))
df_5A['comfort_score'] = df_5A['tokens'].apply(lambda x: sum(1 for w in x if w in comfort_keywords))

# create a tf-idf vectorizer object
tfidf_vectorizer = TfidfVectorizer()

# create tf-idf matrix, referring to the TF-IDF scores of each term
tfidf_matrix = tfidf_vectorizer.fit_transform(df_5A['text'])

# define a function that takes in a user's preferences and returns the top 10 recommended cars
def recommend_cars(user_input):
    n = 10 # number of recommended cars to return
    
    if user_input == 'Safety':
        safety_score= 5 #example of user input
        
        # create a profile based on the user's preferences
        profile_safety = ' '.join([' '.join(safety_keywords) + ' ']*safety_score)
        
        # create a tf-idf matrix for the user's profile
        profile_matrix_safety = tfidf_vectorizer.transform([profile_safety])
        
        # calculate cosine similarities between the user's profile and all cars in the dataset
        cosine_similarities = cosine_similarity(profile_matrix_safety, tfidf_matrix).flatten()
        
        # sort the similarities in descending order
        similar_indices = cosine_similarities.argsort()[::-1]
        
        # get the top 10 recommended cars with their model names and cosine similarity scores
        top_similar_indices = similar_indices[:n]
        recommended_cars = [(df_5A.iloc[index]['model'], cosine_similarities[index]) for index in top_similar_indices]
        
        # filter cars based on type and age
        suv_pickup_cars = df_5A[(df_5A['types_SUV'] == 1) | (df_5A['types_Truck'] == 1)]
        filtered_cars_safety = suv_pickup_cars.sort_values(['age_of_car', 'safety_score'], ascending=[True, False]).head(50)
        
        # calculate cosine similarities between the user's profile and SUV and pickup cars
        cosine_similarities = cosine_similarity(profile_matrix_safety, tfidf_matrix[filtered_cars_safety.index])
        
        # get the indices of the top 10 similar cars
        similar_indices = cosine_similarities.argsort()[0][::-1][:n]
        
        # get the model names of the top 10 similar cars
        top_cars_safety = suv_pickup_cars.iloc[similar_indices]['model'].tolist()
        
        print("Recommended cars based on safety:")
        print(recommended_cars)
        print("Top 5 Safe cars:")
        print(top_cars_safety)
        for i in similar_indices:
            print("- Model:", df_5A['model'][i], "(Cosine Similarity:", cosine_similarities[0][i], ")","CarID:", df_5A['car_id'][i])


    elif user_input == 'Comfort':
        user_comfort_score= 5 #example of user input
        
        # create a profile based on the user's preferences
        profile_comfort= ' '.join([' '.join(comfort_keywords) + ' ']*user_comfort_score)
        
        # create a tf-idf matrix for the user's profile
        profile_matrix_comfort = tfidf_vectorizer.transform([profile_comfort])
        
        # calculate cosine similarities between the user's profile and all cars in the dataset
        cosine_similarities = cosine_similarity(profile_matrix_comfort, tfidf_matrix).flatten()
        
        # sort the similarities in descending order
        similar_indices = cosine_similarities.argsort()[::-1]
        
        # get the top 10 recommended cars with their model names and cosine similarity scores
        top_similar_indices = similar_indices[:n]
        recommended_cars_comfort = [(df_5A.iloc[index]['model'], cosine_similarities[index]) for index in top_similar_indices]
        
        # filter cars based on type and age
        luxury_auto_cars = df_5A[(df_5A['types_Luxury Sedan'] == 1) & (df_5A['transmission_Auto'] == 1)]
        luxury_auto_cars = luxury_auto_cars[luxury_auto_cars['age_of_car'] < 5]
        filtered_cars_comfort = luxury_auto_cars.sort_values(['comfort_score'], ascending=[True]).head(50)  
         
        # calculate cosine similarities between the user's profile and SUV and pickup cars
        cosine_similarities = cosine_similarity(profile_matrix_comfort, tfidf_matrix[filtered_cars_comfort.index])
        
        # get the indices of the top 10 similar cars
        similar_indices = cosine_similarities.argsort()[0][::-1][:n]
        
        # get the model names of the top 10 similar cars
        top_cars_comfort = luxury_auto_cars.iloc[similar_indices]['model'].tolist()


        print("Recommended cars based on comfort:")
        print(recommended_cars_comfort)
        print("Top 5 Comfort cars:")
        print(top_cars_comfort)
        for i in similar_indices:
            print("- Model:", df_5A['model'][i], "(Cosine Similarity:", cosine_similarities[0][i], ")","CarID:", df_5A['car_id'][i])

    else:
        print("Invalid input. Please select only 1 option.")


recommend_cars('Safety')

# Create the GUI window

# create tkinter window

root = tk.Tk()
root.geometry("500x450")
root.configure(bg='white')
root.title("User selection of Car Preference")

label = tk.Label(root, text="Select your preference:")
label.pack()

# Create a PhotoImage object from a file
image_file = Image.open('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/CSC_Logo.png')
image = ImageTk.PhotoImage(image_file)

# Create a Label widget to display the image
image_label = tk.Label(root, image=image)
image_label.pack()
image_label.image = image  # keep a reference to the image

# Create selection buttons
var = tk.StringVar()

radio_button1 = tk.Radiobutton(root, text="Safety", variable=var, value='Safety', font=("Cambria", 14))
radio_button1.pack()

radio_button2 = tk.Radiobutton(root, text="Comfort", variable=var, value='Comfort', font=("Cambria", 14))
radio_button2.pack()

submit_button = tk.Button(root, text="Submit", command=lambda: [selection(var.get()), root.destroy()], font=("Cambria", 14), fg="white", bg="black")
submit_button.pack()

# run the window loop
root.mainloop()

##End##





  


