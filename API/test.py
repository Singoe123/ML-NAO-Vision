import requests
import os

url = 'http://127.0.0.1:8000/recognize_faces'  # Replace with your API endpoint
image_path = os.path.join('images','marcelo.jpg')

with open(image_path, 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json()['face_names'])