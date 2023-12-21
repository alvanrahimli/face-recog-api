import shutil

import cv2
import face_recognition
import os

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from starlette.responses import JSONResponse
import random


def load_known_faces(people_folder):
    known_face_encodings_dict = {}
    for person_name in os.listdir(people_folder):
        person_folder = os.path.join(people_folder, person_name)
        if os.path.isdir(person_folder):
            known_face_encodings_dict[person_name] = []
            for filename in os.listdir(person_folder):
                image_path = os.path.join(person_folder, filename)
                face_encodings = load_face(image_path)
                if len(face_encodings) > 0:
                    known_face_encodings_dict[person_name].append(face_encodings[0])
    return known_face_encodings_dict


def load_face(img_path):
    image = face_recognition.load_image_file(img_path)
    face_encodings = face_recognition.face_encodings(image)
    return face_encodings


def recognize_faces(image_path, face_encodings) -> list[str]:
    image_name = image_path.split("/")[-1]
    original_image = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2RGB)

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

    people_in_pic = []
    face_locations = face_recognition.face_locations(unknown_image)

    for index, face_encoding in enumerate(unknown_face_encodings):
        name = "N/A"
        for person_name, known_encodings in face_encodings.items():
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                name = person_name
                break

        people_in_pic.append(name)
        (x, y, w, z) = face_locations[index]
        cv2.rectangle(original_image, (z - 10, w + 10), (y + 10, x - 10), (50, 205, 50), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(original_image, name, (z, x - 20), font, 0.7, (500, 500, 500), 2)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite(f"output/{image_name}", original_image)
    return people_in_pic


known_face_encodings = load_known_faces("People/")
print("Learned initial data!")

app = FastAPI()
people_names = []


@app.post("/train")
async def train(name: str, image: UploadFile) -> JSONResponse:
    try:
        if not os.path.exists(f"People/{name}"):
            os.makedirs(f"People/{name}")

        rand = random.randint(100, 999)
        file_name = f"People/{name}/{rand}{os.path.splitext(image.filename)[1]}"
        with open(file_name, "wb") as f:
            shutil.copyfileobj(image.file, f)

        face_encodings = load_face(file_name)
        if len(face_encodings) > 0:
            known_face_encodings[name].append(face_encodings[0])
            return JSONResponse(content={"message": "Trained the model"}, status_code=200)
        else:
            return JSONResponse(content={"error": "No face detected"}, status_code=422)

    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/check-attendance")
async def recognize(file: UploadFile) -> JSONResponse:
    try:
        if not os.path.exists("Inputs"):
            os.makedirs("Inputs")

        rand = random.randint(100, 999)
        file_name = f"Inputs/request-{rand}{os.path.splitext(file.filename)[1]}"
        with open(file_name, "wb") as f:
            shutil.copyfileobj(file.file, f)

        people_in_pic = recognize_faces(file_name, known_face_encodings)
        missing_people = [element for element in people_names if element not in people_in_pic]
        return JSONResponse(content={
            "message": f"Attendance check finished",
            "missingPeople": missing_people
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/set-names")
async def set_names(names: str):
    global people_names
    people_names = str.split(names, ";")
