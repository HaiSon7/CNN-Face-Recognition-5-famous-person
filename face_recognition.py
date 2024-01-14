import cv2
import numpy as np
from tensorflow.keras import models

LABELS = ['Bill Gates', 'Elon Musk', 'Jeff Bezos', 'Mark Zuckerberg', 'Steve Jobs']
face_detection = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)
link_model = r"C:\Python\AI\Project\CNN\CNN_Project_Face_Recognition\face_recognition\model_famous_people_h5"
models = models.load_model(link_model)

while True:
    OK , frame = cam.read()
    faces = face_detection.detectMultiScale(frame,1.05,5)
    for (x,y,w,h) in faces:
        roi = cv2.resize(frame[y+2:y+h -2 ,x+2:x+w-2],(70,70))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        p = models.predict(roi.reshape(-1, 70, 70,1))
        print(p)
        print(np.argmax(p))
        print(LABELS[np.argmax(p)])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame,LABELS[np.argmax(p)],(x,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color=(255,255,255),thickness=2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()