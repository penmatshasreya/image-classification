# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:34:23 2020

@author: SNEHIL
"""

import face_recognition as fr
import os
import cv2
import shutil

def Facerec():
    kwn_face_dir= r"C:\Users\SNEHIL\Desktop\Self Study\known faces"
    ukwn_face_dir= r"C:\Users\SNEHIL\Desktop\Self Study\unknown faces"
    copyd = r"C:\Users\SNEHIL\Desktop\Self Study\Processed faces"
    tol = 0.6
    mod = "HAAR"

    known_faces = []
    known_names = []
    for name in os.listdir(kwn_face_dir):
        for filename in os.listdir(f"{kwn_face_dir}/{name}"):
            image = fr.load_image_file(f"{kwn_face_dir}/{name}/{filename}")
            encoding = fr.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)

    filename = os.listdir(ukwn_face_dir)
    image = fr.load_image_file(f"{ukwn_face_dir}/{filename}")
    locations = fr.face_locations(image, model=mod)
    encodings = fr.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dest = shutil.copyfile(f"{ukwn_face_dir}/{filename}", f"{copyd}/{filename}")
    os.remove(f"{ukwn_face_dir}/{filename}")
    for face_encoding, face_location in zip(encodings, locations):
        results = fr.compare_faces(known_faces, face_encoding,tol)
        match = None
        if (True in results):
            match = known_names[results.index(True)]
            return "Snehil"
        else:
            return "Unknown"
    return "ABC"
            
