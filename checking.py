import cv2
import numpy as np
import pygame
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

pygame.mixer.init()
music_directory = r"C:\Users\prana\Videos\music for edits"
music_files = [f for f in os.listdir(music_directory) if f.endswith('.mp3')]

music_index = 0
music_paused = False

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        if not music_paused:
            pygame.mixer.music.pause()
            music_paused = True
            print("No face detected -- Music Paused")
    else:
        if music_paused:
            pygame.mixer.music.unpause()
            music_paused = False
            print("Face detected -- Music Playing")
        elif not pygame.mixer.music.get_busy():
            if music_index >= len(music_files):
                music_index = 0
            pygame.mixer.music.load(os.path.join(music_directory, music_files[music_index]))
            pygame.mixer.music.play()
            music_index += 1

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30)&0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
