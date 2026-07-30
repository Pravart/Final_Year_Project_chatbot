import requests

url = "http://127.0.0.1:5000/chat"

response = requests.post(
    url,
    json={"message": "I failed my exam and I feel hopeless."}
)

print(response.json())
