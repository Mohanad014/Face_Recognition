import cv2
import face_recognition
 
  
#loading the image [import image, eg: Elon Musk] 
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
#image only accepts RGB so convert BGR to RGB 
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#loading or importing the test image 
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
#image only accepts RGB so convert BGR to RGB 
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) 
 
 
#detect the face distance from the camera, [0]-first element
faceLoc = face_recognition.face_locations(imgElon)[0]
#detection of the image, [0]-first element
encodeElon = face_recognition.face_encodings(imgElon)[0]
#draw it on the face, Square like structure, (255,0,255)-Colour of square
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#Testing the face for imgTest, which is located properly or not
faceLocTest = face_recognition.face_locations(imgTest)[0]
#detection of image
encodeTest = face_recognition.face_encodings(imgTest)[0]
#draw it on the face, Square like structure, (255,0,255)-Colour of square
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#using Linear SVM find the image is same or not, comparing encodeElon with encodeTest to test the result
results = face_recognition.compare_faces([encodeElon],encodeTest)
#find the best match, we find the distance, List of images is taken, here only encodeElon
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
#if we print the result,if equal prints true, face distance-[0.45]
print(results,faceDis)
#put text on the image, rounds 2 decimal, origin(50,50), font(Hershey), scale,colour,thickness
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#show the original image
cv2.imshow('Elon Musk',imgElon)
#show the test image
cv2.imshow('Elon Test',imgTest)
#wait key to wait 
cv2.waitKey(0)
