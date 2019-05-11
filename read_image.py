import cv2
import numpy as np
import time

image4 = cv2.imread(r'C:\Users\soma\Pictures\Saved Pictures\jackofhearts.jpg',1)

#cv2.imshow("img",image4)

cap=cv2.VideoCapture(0)

#im3=np.ones((480,640,3))

#w,h=image4.shape[:2]

#print(str(w)+" "+str(h))

while(1):
    ret,frame = cap.read()
    w1,h1=frame.shape[:2]
    #cv2.imshow("img",img2_fg)
    
    #print(str(w1)+" "+str(h1))
    #print("w1h1")
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_red = np.array([100,20,20])
    upper_red = np.array([115,255,255])

    mask = cv2.inRange(hsv,lower_red,upper_red)
    #cv2.imshow('im2',mask);

    kernelOpen=np.ones((7,7))
    kernelClose=np.ones((20,20))

    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    #cv2.imshow('im2',maskFinal);

    cnts,hierarchy=cv2.findContours(maskFinal,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    max1=0
    d=0
    rect=[]

    for c in range(0,len(cnts)-1):
      if cv2.contourArea(cnts[c])>max1:
        max1=cv2.contourArea(cnts[c])
        perimeter = cv2.arcLength(cnts[c],True)
        radius=perimeter/(2*3.14)
        d=c;
        maxp=perimeter


    M = cv2.moments(cnts[d])
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    #approx = cv2.approxPolyDP(d, 0.02 * maxp, True)
    (x, y, w, h)= cv2.boundingRect(cnts[d])
    rect = (x, y, w, h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1);
    cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('im2',frame);
    

    

    print(str(cX)+" "+str(cY))
    print("D")
    res=np.ones((480,640,3))
    res=cv2.resize(image4,(w,h),interpolation=cv2.INTER_LINEAR)
    
    #print(str(w)+" "+str(h))
    time.sleep(1)
    frame[y:y+h,x:x+w]=res
    cv2.imshow('im2',frame);

    k=cv2.waitKey(1)&0xFF
    
    if k==27:
        cv2.destroyAllWindows()
        cap.release()
        break
