# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:53:27 2023

@author: kwanyick, Ivan
"""

# Purpose: For cleaning data & build a recommender system

import pandas as pd

#Load the CSV file
df = pd.read_csv('/Users/ivanong/Documents/GitHub/CarSmartConsultancy/sgcarmart_usedcar_info.csv')
                        
#preview the csv file
print(df.describe())
print(df.info())
print(df.head())





      
