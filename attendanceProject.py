import cv2
import numpy as np
import face_recognition
   
#to import the path
import os
from datetime import datetime
# from PIL import ImageGrab
 #import the ImagesAttendance  
path = 'ImagesAttendance' 
#images will be stored here 
images = []
#image names will be stored here
classNames = [] 
#grab the images list from the ImagesAttendance folder
myList = os.listdir(path)
#output-['mohan.jpg,vikram.jpg,lingesh.jpg)
print(myList)
#read the image for the file, cl is the name of the image
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
 #image will store here
    images.append(curImg)
 #class name will store here with only mohan not mohan.jpg
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
#find the encodings for the image
def findEncodings(images):
    encodeList = []
    for img in images:
     #convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     #find the encodings and append
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#marking attendance, name and time
def markAttendance(name):
 #import attendance file, opening the file,(r+ means read and write)
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
         #split the name and append in nameList
            entry = line.split(',')
            nameList.append(entry[0])
         #to check whether name is already available or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
         #print name and time 
            f.writelines(f'\n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#initialize the webcam
cap = cv2.VideoCapture(0)

#loop to find each image
while True:
 #read the image
    success, img = cap.read()
 #reduce the size of the image
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
 #convert BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 #multiple images will be there, so find the image distance
    facesCurFrame = face_recognition.face_locations(imgS)
 #find encoding of the webcam, sending image and Curframe
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

 #finding the matches, iterate
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
     #matching the faces from encodeListKnown(List),encodeFace(first image)
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
     #compare the distance from encodeListKnown(List),encodeFace(first image)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
     #find the minimum distance 
        matchIndex = np.argmin(faceDis)

     #bounding box around them and write the name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
         #Where to draw the rectangele is done by faceLoc
            y1,x2,y2,x1 = faceLoc
         #hence we reduced the face diatance, we retriving the original face
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
         #draw rectangle 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
         #draw small rectangle to write the name inside
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
         #calls the function and gives the name and store it in CSV(comma seperatd value)
            markAttendance(name)

 #show the image
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
