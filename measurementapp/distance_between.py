import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
prev = cv2.imread('nishant.jpg',1)
# ret, prev = cap.read()
new = cv2.flip(prev, 1)
total_height = 168
gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
        face = cv2.rectangle(prev, (x,y), (x+w,y+h), (255,0,0), 2)
        print("this is my face co-ordinates",face)
        dst = 6421 / w
        dst = '%.2f' %dst
        font = cv2.FONT_HERSHEY_SIMPLEX
        distance = float(dst)+15
        distance = '{0:.2f}'.format(distance)
        cv2.putText(prev, "{}cms".format(distance), (x, y-10), font, 1, (0, 50, 250), 1, cv2.LINE_AA)
diff = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
diff = cv2.blur(diff, (5, 5))
ret, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
threh = cv2.dilate(thresh, None, 3)
thresh = cv2.erode(thresh, np.ones((4, 4)), 1)
fincontours, contor = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.circle(prev, (20, 200), 5, (0, 0, 255), -1)

for contors in fincontours:
        if cv2.contourArea(contors) > 3000:
            (x, y, w, h) = cv2.boundingRect(contors)
            (x1, y1), rad = cv2.minEnclosingCircle(contors)
            x1 = int(x1)
            y1 = int(y1)
            twoD_distance = int((np.sqrt((x1 - 20) ** 2 + (y1 - 200) ** 2))*2.54/96)
            twoD_distance = (55/170)*total_height
            twoD_distance = '{0:.2f}'.format(twoD_distance)
            print(twoD_distance)
            cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(prev, (x1, y1), 5, (0, 0, 255), -1)
arm_size = cv2.putText(prev, "arm {}cms".format(twoD_distance), (0, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
print(arm_size)
prev=cv2.resize(prev, (650,650))
cv2.imshow("Original", prev)

prev = new
_, new = cap.read()
new = cv2.flip(new, 1)

cv2.waitKey(0)


cv2.destroyAllWindows()