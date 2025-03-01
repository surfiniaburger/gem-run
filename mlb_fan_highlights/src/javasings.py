import requests

url = "http://localhost:8080/api/v1/podcast"
payload = {
    "team": "Dodgers",
    "players": ["Shohei Ohtani", "Mookie Betts", "Freddie Freeman"],
    "timeframe": "Last game",
    "game_type": "Regular Season",
    "language": "english",
    "opponent": "Giants"
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())