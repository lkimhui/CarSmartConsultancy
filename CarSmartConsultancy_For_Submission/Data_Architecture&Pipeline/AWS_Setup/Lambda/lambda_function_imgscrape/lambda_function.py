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
from PIL import Image
from io import BytesIO

#naming s3 as variable to access AWS S3
s3 = boto3.client('s3')
bucket = 'sgcarmart-webscrape-data'
key = 'carModelUrls_map/carModelUrls_map.json'
obj = s3.get_object(Bucket=bucket, Key=key)
s3u = boto3.resource('s3')
bucket1 = 'sgcarmart-webscrape-imgdata'
key1 = ''

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

# main function 2: get the image link
def scrape_image(url):
    html = req.get(url)
    soup = BeautifulSoup(html.content, 'lxml')
    
    image_url = ''
    try:
        image_url = soup.body.find('div', {'id':'contentblank'}).select('div:nth-of-type(2)')[0].select('div:nth-of-type(5)')[0].select('div:nth-of-type(1)')[0].next_sibling.next_sibling.findAll('img')[0].attrs['src']
    except:
        image_url = None
        pass
    
    model = soup.body.find('div', {'id':'contentblank'}).select('div:nth-of-type(2)')[0].select('div:nth-of-type(1)')[0].text.strip()
    #clean the model text
    if model.find("(") >= 0:
        model = model.partition("(")[0]
    
    return model, image_url

# subfunction 1
def download_image(model, url):
    # use pillow to read the btye string
    response = req.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        #specify your directory below
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Upload the image to S3
        imgname = f"{model}.jpg"
        image_bytes = BytesIO()
        img.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        
        s3_object = s3u.Object(bucket1, imgname)
        s3_object.put(Body=image_bytes)

# main function 3
def save_image(df):
    return np.where(df['Image'] != None, download_image(df['Model'], df['Image']), None)


def lambda_handler(event, context):
    # run main function 1: get webpage url based o
    carModelUrls = scrape_url(1)
    images = list(map(scrape_image, carModelUrls))

    # overwrite map dict
    mapname = f"/tmp/carModelUrls_map.json"
    with open(mapname, 'w') as mapfile:
        json.dump(cur_carModelUrls_map, mapfile, indent=1)
    s3u.Bucket(bucket).upload_file(mapname, key)

    # convert the model and image link to a dataframe
    df = pd.DataFrame(images, columns=['Model', 'Image'])

    # run main function 3: download image to directory
    df.apply(save_image, axis=1)