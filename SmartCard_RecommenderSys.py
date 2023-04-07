# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

# Purpose: For cleaning data & build a recommender system

import pandas as pd

#Load the CSV file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmart/df_all.csv')

#preview the csv file
print(df.describe())
print(df.info())
print(df.head())

# clean punctuations 

# Remove punctuation
df['coe'] = df['coe'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Try Cross Tabulation (Car type vs. Power)
pd.crosstab(df['type'],df['power'])


<<<<<<< Updated upstream
=======
print("Smart smart smart")
print('hi kim huiiiiiii')
print('HI IVAN!')
>>>>>>> Stashed changes



      
