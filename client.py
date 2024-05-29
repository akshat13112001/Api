import requests

url = "http://localhost:8000/generate"
data = {"prompt": "Essay on ai"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print(response.json()) 
else:
    print(f"Request failed with status code: {response.status_code}")