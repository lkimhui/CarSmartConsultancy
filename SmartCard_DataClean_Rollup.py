# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

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
import datetime

##To note before running code: To change path for ROW 25, 176, 238##
#Load the CSV file (To add your local Github folder path)
#df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
df = pd.read_csv('/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/sgcarmart_usedcar_info.csv')
    
#preview the csv file
print(df.describe())
print(df.info())
print(df.head())

#Dropping all rows which has N.A (Considering only used car entries with complete data)
df_nonull = df.dropna()
# Print the number of rows before and after dropping nulls
print('Number of rows before dropping nulls:', len(df))
print('Number of rows after dropping nulls:', len(df_nonull))
print(df_nonull.head())

df_alllesstext= df_nonull
print(df_alllesstext.head())
print(df_alllesstext['price'].head()) #quick check on specific column

#after initial recommender 1, it seems that an extensive data cleaning is require. Data cleaning for each column:
#price
df_alllesstext['price'] = df_alllesstext['price'].str.replace(',', '')
df_alllesstext['price'] = df_alllesstext['price'].str.replace('$', '')
df_alllesstext['price'] = df_alllesstext['price'].replace('N.A.', np.nan)
df_alllesstext['price'] = df_alllesstext['price'].replace('N.A', np.nan)
df_alllesstext['price'] = df_alllesstext['price'].astype(float)
    
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
df_alllesstext['curb_weight'] = df_alllesstext['curb_weight'].astype(float)

#power
df_alllesstext['power'] = df_alllesstext['power'].str.replace(r'\(.*?\)', '', regex=True)
df_alllesstext['power'] = df_alllesstext['power'].str.replace(',', '')
df_alllesstext['power'] = df_alllesstext['power'].str.replace('kW', '')
df_alllesstext['power'] = df_alllesstext['power'].replace('N.A.', np.nan)
df_alllesstext = df_alllesstext[~df_alllesstext['power'].str.contains('More than 6')]
df_alllesstext['power'] = df_alllesstext['power'].astype(float)

#registration_dates (derived: age of car)
# convert the 'registration_date' column to a datetime object
df_alllesstext['registration_date'] = pd.to_datetime(df_alllesstext['registration_date'], format='%d-%b-%y')
# calculate the difference between the registration date and today's date
today = datetime.datetime.now()
df_alllesstext['age_of_car'] = (today - df_alllesstext['registration_date']) / pd.Timedelta(days=365)
# show the updated DataFrame
print(df_alllesstext)

#manufacturered year (derived: year since launch)
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
# drop the original 'number_of_owner' column
df_alllesstext = df_alllesstext.drop('number_of_owner', axis=1)

#type
# use get_dummies() to one-hot encode the 'type' column
df_onehot = pd.get_dummies(df_alllesstext['type'], prefix='type')
# concatenate the one-hot encoded features with the original DataFrame
df_alllesstext = pd.concat([df_alllesstext, df_onehot], axis=1)
# drop the original 'type' column
df_alllesstext = df_alllesstext.drop('type', axis=1)

df_alllesstext['model'] = df_alllesstext['model'].str.replace(r'\(.*?\)', '', regex=True)
df_alllesstext['model'].unique()

# show the updated DataFrame
print(df_alllesstext)

df_alllesstext.info()

df_alllesstext.dropna(inplace=True)

null_counts = df_alllesstext.isnull().sum()
print(null_counts)

# Generate timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#This is for data checking after cleaning
file_name = f"datachecking_{timestamp}.xlsx"
#file_path = '/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
file_path = '/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
df_alllesstext.to_excel(file_path + file_name, index=False)

##Data Roll up to reduce sparsity##

#selected 26 out of 41 columns
selected_cols = ['model', 'price', 'depreciation', 'mileage', 'road_tax', 'deregistration_value', 
                 'coe', 'curb_weight', 'power', 'car_features', 'car_accessories', 'descriptions', 
                 'age_of_car', 'years_since_launch', 'trasmission_Auto', 
                 'trasmission_Manual', 'number_of_owner_More than 6',
                 'type_Hatchback', 'type_Luxury Sedan', 'type_MPV', 'type_Mid-Sized Sedan',
                 'type_SUV', 'type_Sports Car','type_Stationwagon','type_Truck','type_Van']
df_selected = df_alllesstext[selected_cols]
print(df_selected.head())

df_rollup = df_selected.groupby('model').agg({
    'price': 'median',
    'depreciation': 'median',
    'mileage': 'median',
    'road_tax': 'median',
    'deregistration_value': 'median',
    'coe': 'median',
    'curb_weight': 'median',
    'power': 'median',
    'car_features': 'sum',
    'car_accessories': 'sum',
    'descriptions': 'sum',
    'age_of_car': 'median',
    'years_since_launch': 'median',
    'trasmission_Auto': 'max',
    'trasmission_Manual': 'max',
    'number_of_owner_More than 6': 'max',
    'type_Hatchback': 'max',
    'type_Luxury Sedan': 'max',
    'type_MPV': 'max',
    'type_Mid-Sized Sedan': 'max',
    'type_Hatchback': 'max',
    'type_SUV': 'max',
    'type_Sports Car': 'max',
    'type_Stationwagon': 'max',
    'type_Truck': 'max',
    'type_Van': 'max',
    })

print(df_rollup)

#Add unique ID(new column)
#Generate a sequence of integers for the unique IDs
ids = range(1, len(df_rollup)+1)
df_rollup['car_id']= ids

#reset index as groupby function will fix 'model' as first column
df_rollup = df_rollup.reset_index()
# move 'car_id' to the first column
df_rollup.insert(0, 'car_id', df_rollup.pop('car_id'))  
print(df_rollup.head())

# Generate timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

file_name = f"datarollup_{timestamp}.csv"
#file_path = '/Users/ivanong/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
file_path = '/Users/kwanyick/Documents/GitHub/CarSmartConsultancy/data/cleaned_data/'
df_rollup.to_csv(file_path + file_name, index=False)

#Note no normalization applied here

##End of Data Cleaning and Roll up## 


