import requests
import json
import time

API_KEY = "xxxxxxxx"  

locations = [
    "1891 Cyrville Rd, Gloucester, ON K1B 1A9",
    "1555 Alta Vista Dr, Ottawa, ON K1G 0G2",
    "2210 Bank St, Ottawa, ON K1V 1J5",
    "2118 Bank St, Ottawa, ON K1V 8W7",
    "1642 Merivale Rd, Nepean, ON K2G 4A1",
    "3777 Strandherd Dr, Nepean, ON K2J 4B1",
    "700 Eagleson Rd, Kanata, ON K2M 2G9",
    "555 Legget Dr, Ottawa, ON K2K 3B8",
    "2121 Carling Ave, Ottawa, ON K2A 1G9",
    "351 Richmond Rd, Ottawa, ON K2A 0E7",
    "55 ByWard Market Sq, Ottawa",
    "90 Elgin St, Ottawa, ON K1P 5E9",
    "430 Bank St, Ottawa, ON K2P 1Y8",
    "789 Bank St, Ottawa, ON K1S 2J7",
    "2263 Portobello Blvd., Ottawa, ON K4A 0X3",
    "1980 Trim Rd, Orléans, ON K4A 5L5"
]

def get_distance_matrix(locations, api_key):
    matrix = []
    total = len(locations)
    
    for i in range(total):
        origins = locations[i]
        destinations = "|".join(locations)

        url = f"https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": origins,
            "destinations": destinations,
            "mode": "driving",
            "key": api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data["status"] != "OK":
            raise Exception(f"Error in API response: {data}")

        row = []
        for element in data["rows"][0]["elements"]:
            if element["status"] == "OK":
                distance_km = round(element["distance"]["value"] / 1000)
                row.append(distance_km)
            else:
                row.append(-1)

        matrix.append(row)
        time.sleep(1)  

    return matrix

distance_matrix = get_distance_matrix(locations, API_KEY)

print("distance_matrix = [")
for row in distance_matrix:
    print("  " + str(row) + ",")
print("]")
