# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:00:32 2023

@author: kwanyick, Ivan
"""
"CarSmart: Using Great Expectation to validate Input File"

!pip install great_expectations

import pandas as pd
import great_expectations as ge


from google.colab import files
uploaded = files.upload()

# Load data using great_expectations
my_df = ge.read_csv("loans_2007_v1.csv")

# You can use all of pandas’ normal methods on it + some more from great_expectations
my_df.head()

# Check the list of great_expectations methods write "my_df.expect_"
#my_df.expect... 
# autofill method

"""check out the [link ](https://legacy.docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html#expectation-glossary)for more expectation methods

"""

# Check if the number of columns are correct
my_df.expect_table_column_count_to_equal(22)

# Check if the columns have Null values or not
my_df.expect_column_values_to_not_be_null('loan_amnt')

my_df['home_ownership'].unique()

# Check if the column have the desired set of values
# Always expect home ownership only has these input values - RENT/OWN/MORTAGE/OTHER/NONE
my_df.expect_column_values_to_be_in_set('home_ownership', value_set=set(['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE']))

"""Great Expectation starts tracking with "get_expectation"
Different from MLFlow where we need to start.
"""

my_df.get_expectation_suite()

# Not by default
my_df.get_expectation_suite(discard_failed_expectations=False)

"""Writing all the rules into Json file, which can be import to other platforms"""

import json

with open( "my_expectation_file.json", "w") as my_file:
    my_file.write(
        json.dumps(my_df.get_expectation_suite(discard_failed_expectations=False).to_json_dict())
    )