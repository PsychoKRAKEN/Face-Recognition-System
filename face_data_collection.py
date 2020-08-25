# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data



#Import 
import cv2
import numpy as np

#Init Webcam
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]
dataset_path="data/"
file_name=input("Enter the name of person: ")

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
        
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    if len(faces)==0:
        continue
        
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    
    #Pick the last face bcoz it has largest area
    for face in faces[-1:]:
        #draw boundary box 
        
        x,y,w,h=face
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        
        #Extract or crop out region of intrest
        offset=10
        face_section=gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_section))
    
    
    #cv2.imshow("Frame",frame)
    cv2.imshow("gray_frame",gray_frame)
    
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break;
        
        
#convert face data into numpy array
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save the data
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Saved Successfully!!!")

cap.release()
cv2.destroyAllWindows()


