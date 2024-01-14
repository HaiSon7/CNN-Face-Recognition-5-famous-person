import cv2
import numpy as np
import os

face_detection = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
#BGR
cam = cv2.VideoCapture(0)
i = 0
while True:
    OK , frame = cam.read()
    faces = face_detection.detectMultiScale(frame,1.38,5)
    for (x,y,w,h) in faces:
        roi = cv2.resize(frame[y+2:y+h -2 ,x+2:x+w-2],(100,100))

        cv2.imwrite('image/img_faces/Train/HaiSon/img_{}.jpg'.format(i),roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 1)
        i+=1

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()












# path = "image"
#
# for i in os.listdir(path):
#     path_to_j = os.path.join(path,i)
#     for j in os.listdir(path_to_j):
#         if j.endswith(".jpg"):
#             pass









# img_path = r"image/manyface.jpg"
# img_path1 = r"image/friends-photo.jpg"
#
# img = cv2.imread(img_path)
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#

#
# while True:
#     faces = face_detection.detectMultiScale(img_gray,1.3,5)
#     i =0
#     for (x,y,w,h) in faces:
#         face = cv2.resize(img[y:y+h,x:x+w],(32,32))
#         cv2.imwrite('image/img_faces/img_face_{}.jpg'.format(i),face)
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         i+=1
#     cv2.imshow("Image",img)
#     #cv2.imshow("Image_Gray", img_gray)
#     if cv2.waitKey(0):
#         break
#
# cv2.destroyAllWindows()