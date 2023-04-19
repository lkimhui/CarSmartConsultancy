# import libraries
import bs4
from bs4 import BeautifulSoup
import requests as req
import pickle
import re
import pandas as pd
import numpy as np
import json
import boto3

#naming s3 as variable to access AWS S3
s3 = boto3.client('s3')
bucket = 'sgcarmart-webscrape-data'
key = 'id_dl_map/id_dl_map.json'
obj = s3.get_object(Bucket=bucket, Key=key)
s3u = boto3.resource('s3')
key1 = 'usedcar_info/usedcar_info.txt'

# find a spefic href tag pattern
def car_model_href(href):
    return href and re.compile(r"(^(info.php).*)").search(href)
    
# define a function to scrape features from SGCar Mart
def feature_scraping(url):
    print(url)
    # Get the html page
    html_text = req.get(url)
    soup = BeautifulSoup(html_text.content, "lxml")
    text = soup.body
    return text

def lambda_handler(event, context):
    #get data from s3
    json_file = obj['Body'].read().decode('utf-8')
    
    # open id_dl json file and load back as id_dl map dict
    with open('data/id_dl_map.json') as json_file:
        id_dl_map = json.load(json_file)
    
    # creating lists for records
    cur_id_dl_map = {}
    id_list = []
    dl_list = []
    
    # loop through all pages in sgcarmart used car listing
    BRSR = 0
    RPG = 100 #20

    #28th March - completed 0 to 10000 on 28th March
    for BRSR in range(100, 200, 100): 
        url = f"https://www.sgcarmart.com/used_cars/listing.php?BRSR={BRSR}&RPG={RPG}"
        html_text = req.get(url)
        soup = BeautifulSoup(html_text.content, "lxml")
        listings = soup.body.find('div', {'class': 'listing_searchbar_position'}).p.string.replace(" ", "")
        cleaned_listings = re.sub("\D", '', listings)
        listings_count = int(cleaned_listings)
        # car models found per page
        car_model_list = soup.body.find('div', {'id':'content'}).find('form', {'name':'searchform'}).next_sibling.next_sibling.find_all(href=car_model_href, string=True)
        for model in car_model_list:
            id = str(model).partition('ID')[2].partition('&')[0].partition('=')[-1]
            dl = str(model).partition('DL')[2].partition('"')[0].partition('=')[-1]
            if id not in id_dl_map:
                id_dl_map[id] = dl
                cur_id_dl_map[id] = dl
                id_list.append(id)
                dl_list.append(dl)

    # overwrite id_dl json file with latest id_dl map
    with open('/tmp/id_dl_map.json', 'w') as file:
        json.dump(id_dl_map, file, indent=1)

    body = []
    errorUrlMap = {}
    for key, value in cur_id_dl_map.items():
        try:
            url = f"https://www.sgcarmart.com/used_cars/info.php?ID={key}&DL={value}"
            textdata = feature_scraping(url)
            body.append(textdata)
        except:
            errorUrlMap[key] = value
            pass
    
    f = open("/tmp/usedcar_info.txt", "w")
    # write in html body
    f.write(str(body))
    #upload the data into s3
    upload = s3u.Bucket(bucket).upload_file('/tmp/usedcar_info.txt', key1)
