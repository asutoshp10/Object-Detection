import cv2
import numpy as np
import os
import tensorflow as tf
import keras

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     print('it is')
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         print("Using GPU:", gpus[0])
#     except RuntimeError as e:
#         print(e)
model=keras.models.load_model(r'C:\Users\asuto\Desktop\sld\obj_dect_x.h5',)
img=r'C:\Users\asuto\Desktop\AI proj\Yolov1_Cyborg\Dataset\bk_img\bk24.png'
cap=cv2.VideoCapture(r'now.mp4')
# cap=cv2.VideoCapture(1)
# for i in range(4):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Device {i} is available.")
#         cap.release()
#     else:
#         print(f"Device {i} is not available.")
def empty(x):
    pass
cv2.namedWindow('hsv')
cv2.resizeWindow('hsv',640,240)
cv2.createTrackbar('1','hsv',200,255,empty)
cv2.createTrackbar('2','hsv',150,255,empty)

def get_bbox(canny,frame_copy):
    cont,f=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in cont:
        if cv2.contourArea(cnt)>1000:
            # cv2.drawContours(frame_copy,cnt,-1,(0,255,0),5)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x_,y_,w,h=cv2.boundingRect(approx)
            cv2.rectangle(frame_copy,(x_,y_),(x_+w,y_+h),(0,255,0),5)
            p_img=frame_copy[y_:y_+h,x_:x_+w]
            pred=model.predict(np.expand_dims(cv2.resize(p_img,(128,128)),axis=0))
            li=['capacitor','data','IC','resistor','transistor']
            x=li[np.argmax(pred)]
            cv2.putText(frame_copy,x,(y_+h-5,x_),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            # try:
            #     cv2.imshow('p_img',p_img)
            # except Exception as e:
            #     print(e)

while True:
    ret,frame=cap.read()
    print(frame.shape)
    frame_copy=frame.copy()
    # cv2.imshow('frame',cv2.cvtColor(frame,cv2.COLOR_BGR2HSV))
    a=cv2.getTrackbarPos('1','hsv')
    b=cv2.getTrackbarPos('2','hsv')
    # blur=cv2.GaussianBlur(frame,(7,7),1)
    # hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    # blur2=cv2.bilateralFilter(hsv,3,35,35)
    blur=cv2.bilateralFilter(frame,3,35,35)
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    canny=cv2.Canny(blur,a,b)
    kernel=np.ones((5,5))
    dilate=cv2.dilate(canny,kernel,iterations=1)
    get_bbox(dilate,frame_copy)
    cv2.imshow('frame_copy',frame_copy)
    if cv2.waitKey(1) == ord('q'):
        break
