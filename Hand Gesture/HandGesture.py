
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import math
import sys
import os
import random

cap = cv2.VideoCapture(0)
     
while(1):
        
    try: 
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define roi
        roi=frame[100:400, 100:400]
        
        cv2.rectangle(frame,(100,100),(400,400),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # range of skin color in HSV
        lowerskin = np.array([0,20,70], dtype=np.uint8)
        upperskin = np.array([20,255,255], dtype=np.uint8)
        
        #create mask to extract skin color
        mask = cv2.inRange(hsv, lowerskin, upperskin)
        
        #extrapolate 
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        #blur the image using gaussianblur
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        #contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        #convex hull 
        hull = cv2.convexHull(cnt)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
        #calculate area ratio
        arearatio=((areahull-areacnt)/areacnt)*100
    
        #find the defects in convex hull 
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        # k = no. of defects
        k=0
        
        #defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # cosine rule
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 60 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 60 and d>30:
                k += 1
                cv2.circle(roi, far, 3, [0,0,0], -1)
            
            
            
        k+=1
        #path = "D:\\NUS\\ProjectSS\\data\\inventory\\images"
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if k==1:
            if areacnt<2000:
                cv2.putText(frame,'See Layouts',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:    
                getImage = cv2.imread(r'D:\NUS\ProjectSS\data\Structure1.jpg')
                cv2.imshow("Image", getImage)
                #random_file = random.choice(os.listdir(path))
                #path1 = os.path.join(path,random_file)
                #print(path1)
                #getImage = cv2.imread(path1)
                #cv2.imshow("Image", getImage)
                    
        elif k==2:
            getImage = cv2.imread(r'D:\NUS\ProjectSS\data\Structure2.jpg')
            cv2.imshow("Image", getImage)
            
        elif k==3:
            getImage = cv2.imread(r'D:\NUS\ProjectSS\data\Structure3.png')
            cv2.imshow("Image", getImage)
            
        elif k==4:
            getImage = cv2.imread(r'D:\NUS\ProjectSS\data\Structure4.jpg')
            cv2.imshow("Image", getImage)
            
            
        elif k==5:
            #cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            getImage = cv2.imread(r'D:\NUS\ProjectSS\data\FinDesign.jpg')
            cv2.imshow("Image", getImage)
            
            
        else :
            cv2.putText(frame,'These are all designs we have',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()  

