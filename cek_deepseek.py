import requests

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-bf112ef4c6f54c48b327dea735a42bba",
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Apa itu DeepSeek?"}
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
