# import libraries
from bs4 import BeautifulSoup
import requests as req
import pickle
import re
import pandas as pd
import numpy as np
import functools
import operator
import datetime
import json
import boto3

#naming s3 as variable to access AWS S3
s3 = boto3.client('s3')
bucket = 'sgcarmart-webscrape-data'
key = 'carModelUrls_map/carModelUrls_map.json'
obj = s3.get_object(Bucket=bucket, Key=key)
s3u = boto3.resource('s3')
key1 = 'car_html_content/'

# get data from s3
json_file = obj['Body']
# load as map dict
carModelUrls_map = json.load(json_file)
cur_carModelUrls_map = {}

# define a function which scrape url based on number of pages
# subfunction 1
def get_carModelUrl(url):
    html = req.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    href = soup.body.find('div', {'id':'content'}).find('form', {'name':'searchform'}).next_sibling.next_sibling.find_all(href=re.compile(r"(^(info.php).*)"), string=True)
    return href

# subfunction 2
def get_keypairs(hrefString):
    idValue = re.search("(?<=\?ID=)\d+(?=\&)", str(hrefString))
    dlValue = re.search("(?<=\;DL=)\d+(?=(\"|\&))", str(hrefString))
    idValue = idValue[0].strip()
    dlValue = dlValue[0].strip()
    if idValue not in carModelUrls_map:
        cur_carModelUrls_map[idValue] = dlValue
    return idValue, dlValue

# main function 1: get the webpage url
def scrape_url(page):
    #convert page to list of car per page
    pageToList = [(i+1)*100 for i in range(page)]
    #create BeautifulSoup object
    urlList = [f"https://www.sgcarmart.com/used_cars/listing.php?BRSR={car}&RPG=100" for car in pageToList]
    carModelUrlList = list(map(get_carModelUrl, urlList))
    #flatten the list
    carModelUrlList = functools.reduce(operator.iconcat, carModelUrlList, [])
    #get keypairs(id-dl)
    keypairsList = list(map(get_keypairs, carModelUrlList))
    print(keypairsList)
    #reformat the webpage url
    carModelUrlList = list(map(lambda x: f"https://www.sgcarmart.com/used_cars/info.php?ID={x[0]}&DL={x[1]}", keypairsList))
    return carModelUrlList

# define a function to scrape features from SGCar Mart
def feature_scraping(url):
    print(url)
    # Get the html page
    html = req.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    filename = f"{key1}car_html_content_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
    # traverse paragraphs from soup
    s3_object = s3u.Object(bucket, filename)
    s3_object.put(Body=bytes(str(soup), encoding='utf-8'))
    return soup
    
def lambda_handler(event, context):

    # run main function 1: get webpage url based o
    carModelUrlList = scrape_url(5)

    # overwrite map dict
    mapname = f"/tmp/carModelUrls_map.json"
    with open(mapname, 'w') as mapfile:
        json.dump(cur_carModelUrls_map, mapfile, indent=1)
    s3u.Bucket(bucket).upload_file(mapname, key)

    features = list(map(feature_scraping, carModelUrlList))