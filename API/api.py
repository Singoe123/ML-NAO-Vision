from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import cv2
import face_recognition
import numpy as np
import shutil
import os

known_face_encodings = []
known_face_names = []

known_images = [] 
known_names = []

image_folder = "images"
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        known_images.append(filename)
        known_names.append(os.path.splitext(filename)[0])

def encode_faces():
    for image_path, name in zip(known_images, known_names):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    encode_faces()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/test")
async def read_test():
    return {"message": "test"}

@app.post("/recognize_faces/")
async def recognize_faces(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(temp_file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    os.remove(temp_file_path)

    return {"face_names": face_names}