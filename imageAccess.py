from pprint import pprint
import googlemaps
import requests
import os
import random

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
streetview_url = "https://maps.googleapis.com/maps/api/streetview"
metadata_url   = "https://maps.googleapis.com/maps/api/streetview/metadata"

gmaps = googlemaps.Client(API_KEY)

STREETVIEW_COUNTRIES = [

    # Europe
    "Albania","Andorra","Austria","Belgium","Bulgaria","Croatia",
    "Czech Republic","Denmark","Estonia","Finland","France",
    "Germany","Greece","Hungary","Iceland","Ireland","Italy",
    "Latvia","Liechtenstein","Lithuania","Luxembourg","Malta",
    "Monaco","Montenegro","Netherlands","North Macedonia",
    "Norway","Poland","Portugal","Romania","Serbia",
    "Slovakia","Slovenia","Spain","Sweden","Switzerland",
    "United Kingdom",

    # Asia
    "Bangladesh","Cambodia","Hong Kong","India","Indonesia",
    "Israel","Japan","Kyrgyzstan","Malaysia",
    "Mongolia","Nepal","Pakistan","Philippines","Singapore",
    "South Korea","Sri Lanka","Taiwan","Thailand","Vietnam","Bhutan",

    # Americas
    "Canada","Mexico","United States",
    "Argentina","Bolivia","Brazil","Chile","Colombia",
    "Ecuador","Peru","Uruguay",

    # Oceania
    "Australia","New Zealand",

    # Africa
    "Botswana","Eswatini","Ghana","Kenya","Lesotho",
    "Nigeria","Rwanda","Senegal","South Africa","Tunisia"
]

###Use Kaggle DataSet as inspiration
for country in STREETVIEW_COUNTRIES:
    print(country)
    try:
        os.mkdir(country)
    except:
        print("Directory already exists.")
    os.chdir(country)

    if len(os.listdir()) >= 100:
        os.chdir("..")
        continue

    countryGeoCode = gmaps.geocode(country)
    bounds = countryGeoCode[0]["geometry"].get("bounds") or (countryGeoCode[0]["geometry"])["viewport"]

    min_lat = bounds["southwest"]["lat"]
    min_lng = bounds["southwest"]["lng"]
    max_lat = bounds["northeast"]["lat"]
    max_lng = bounds["northeast"]["lng"]

    #print(min_lat, min_lng, max_lat, max_lng)

    saved = len(os.listdir())
    while saved < 100:
        ran_lat = random.uniform(min_lat, max_lat)
        ran_lng = random.uniform(min_lng, max_lng)
        #print(ran_lat, ran_lng)

        meta_params = {
            "location" : f"{ran_lat}, {ran_lng}",
            "key": API_KEY
        }

        metaResponse = requests.get(metadata_url, params = meta_params)
        meta = metaResponse.json()
        if meta.get("status") != "OK":
            continue

        params = {
            "size": "640x640",
            "location": f"{ran_lat},{ran_lng}",
            "heading": random.uniform(0,360),
            "pitch": 0,
            "fov": 90,
            "key": API_KEY
        }

        response = requests.get(streetview_url, params = params)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            filename = f"img_{saved:03d}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
                print(saved, "Uploaded")
                saved+=1
    os.chdir("..")
    
