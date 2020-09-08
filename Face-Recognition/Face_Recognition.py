# Creator : Lissan Koirala
# Date Of Creation : 19/02/2020

print("\n\n\n")
print("-"*119)
print("Welcome to the Face Recognition")
print("-"*119)

# Imports
import os
import cv2 
import face_recognition
import numpy as np
from time import sleep
import pickle

# Get the encodings of the recognised faces
def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

# Classifying the recognised face to the captured face
def classify_face(faces,im):
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(im, 1)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        os.remove("captured_face.jpg")
        return face_names

# Taking a photo of the user
def take_photo(name):
    if ".png" in name or ".jpg" in name:
        file_name = name
    else:
        file_name = "captured_face.jpg"
    x = 0
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        return "Error"
    cv2.imwrite(file_name, frame)
    if x != "fail":
      return "Sucess"

# Function for recognising the existing user
def existing():
    print("-"*119)
    print("Now your real time photo will be taken to be Recognised, LookUp at the Webcam")
    print("-"*119)
    take_photo("default")
    print("Your Photo has been taken, now it will recognised!")
    print("-"*119)
    faces = get_encoded_faces()
    name = classify_face(faces, "captured_face.jpg")
    print("It's you : " + name[0])
    print("-"*119)
    return name[0]

# Function for adding a new user
def new():
    print("-"*119)
    name = input("Insert your Identity : ") + ".jpg"
    print("-"*119)
    print(name.replace(".jpg",'') + ", Your photo is being taken, that will be your Identity! LookUp on the Webcam!")
    print("-"*119)
    os.chdir("D:/Programming/Python - Lissan/Face_recogniton/faces")
    take_photo(name)
    print("You are now a new Member!\n"+"-"*119 + "\nNow I can recognise you by the name : " + name.replace(".jpg",''))
    print("-"*119)
    get_encoded_faces()

# Initial user inteface
print("Choose One Of the following!")
print("1. Insert a New User")
print("2. Recognise a Exixting User")
print("-"*119)
what = str(input("What you want to do? : "))
if "1" in what or "i" in what:
    new()
else:
    name = existing()

input() # So that the program doesn't exists










