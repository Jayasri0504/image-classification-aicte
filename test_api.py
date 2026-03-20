import requests

url = "https://jayasri0509-waste-classifier-model.hf.space/predict"

files = {
    "file": open("food.jpg", "rb")   # put any image here
}

response = requests.post(url, files=files)

print(response.json())