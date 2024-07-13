
import numpy as np

import cv2
from imutils import paths
import numpy as np
import imutils
import cv2

facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

cap.set(3,640) # set Width

cap.set(4,480) # set Height
boxes = []
while True:

   ret, img = cap.read()

   img = cv2.flip(img, 1)

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   edged = cv2.Canny(gray, 35, 125)
 
   faces = facecascade.detectMultiScale(

       gray,
       
       
       scaleFactor=1.2,

       minNeighbors=5,    

       minSize=(20, 20)

   )

   for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        count = len(faces)
        roi_color = img[y:y+h, x:x+w] 
        y = y + h
        print('y=',y)
        
        x = x + w
        print('x=',x)
        cv2.putText(img,'Y=' + str(y) , (img.shape[1] - 600, img.shape[0] - 50) , cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.putText(img, 'X=' +  str(x) ,(img.shape[1] - 600, img.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        
        length = roi_gray.shape[0]
        breadth = roi_gray.shape[1]
        
        
        
        cv2.putText(img,'Object length =' + str(length) , (img.shape[1] - 600, img.shape[0] - 20) , cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        #cv2.putText(img, 'breadth =' +  str(breadth) ,(img.shape[1] - 700, img.shape[0] - 400), cv2.FONT_HERSHEY_SIMPLEX,,0.7, (0, 255, 0), 2)
        
        focalLength = 3.85*100
        
        Distance = (length * breadth)/focalLength
        cv2.putText(img,'Distance =' + str(Distance) ,(img.shape[1] - 200, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 3)
        cv2.putText(img,'person count =' + str(count) ,(img.shape[1] - 200, img.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 3)
        D = Distance
        if D<10:
            s="Person Detected distance approximatly on 104 cm"
        elif D>10 and D<13:
            s="Person Detected distance approximatly on 99 cm"
        elif D>13 and D<15:
            s="Person Detected distance approximatly on 94 cm"
        elif D>15 and D<17:
            s="Person Detected distance approximatly on 84 cm"
        elif D>17 and D<20:
            s="Person Detected distance approximatly on 75 cm"
        elif D>20 and D<25:
            s="Person Detected distance approximatly on 67 cm"
        elif D>25 and D<30:
            s="Person Detected distance approximatly on 59 cm"
        elif D>30 and D<35:
            s="Person Detected distance approximatly on 51 cm"
        elif D>35 and D<40:
            s="Person Detected distance approximatly on 46 cm"
        elif D>40 and D<60:
            s="Person Detected distance approximatly on 40 cm"
        elif D>60 and D<70:
            s="Person Detected distance approximatly on 38 cm"
        elif D>70 and D<80:
            s="Person Detected distance approximatly on 35 cm"
        elif D>80 and D<90:
            s="Person Detected distance approximatly on 34 cm"
        elif D>90 and D<100:
            s="Person Detected distance approximatly on 32 cm"
        elif D>100 and D<120:
            s="Person Detected distance approximatly on 31 cm"
        elif D>120 and D<150:
            s="Person Detected distance approximatly on 30 cm"
        elif D>150 and D<170:
            s="Person Detected distance approximatly on 27 cm"
        elif D>170 and D<190:
            s="Person Detected distance approximatly on 24 cm"
        elif D>190 and D<210:
            s="Person Detected distance approximatly on 21 cm"
        elif D>210 and D<230:
            s="Person Detected distance approximatly on 19 cm"
        elif D>230 and D<250:
            s="Person Detected distance approximatly on 17 cm"
        elif D>250 and D<270:
            s="Person Detected distance approximatly on 14 cm"
        elif D>270 and D<290:
            s="Person Detected distance approximatly on 11 cm"
        elif D>290:
            s="Person Detected distance approximatly on 9 cm or less"
            
        A = s
        #print(A)
        cv2.putText(img, str(A) ,(img.shape[1] - 630, img.shape[0] - 400), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        import datetime
        now = datetime.datetime.now()
        B=now.strftime("%Y-%m-%d %H:%M:%S")
        #print ("Current date and time : ")
        #print (B)
        cv2.putText(img, str(B) ,(img.shape[1] - 630, img.shape[0] - 450), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2)
        #cv2.putText(frame,TEXT ON VIDEO',(50, 50),font, 1,(0, 255, 255),2,cv2.LINE_4) 
        cv2.imshow('video',img)

   k = cv2.waitKey(30) & 0xff

   if k == 27: # press ‘ESC’ to quit

       break

cap.release()

cv2.destroyAllWindows()