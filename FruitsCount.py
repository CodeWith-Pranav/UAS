#importing libraries
import cv2
import numpy as np
from ultralytics import YOLO

#training yolo model on pretrained data set 
model=YOLO(r'C:\Users\deepa\Downloads\best.pt')
results=model.predict(r'C:\Users\deepa\OneDrive\Pictures\Saved Pictures\Screenshots\Screenshot 2025-02-03 172929.png',save=True)
results2=model.predict(r'C:\Users\deepa\OneDrive\Pictures\Saved Pictures\Screenshots\Screenshot 2025-02-03 172941.png',save=True)

#getting numpy array containing coordinates and classifications
xyxy = results[0].boxes.xyxy.cpu().numpy()
classpred = results[0].boxes.cls.cpu().numpy()
arr = np.column_stack((xyxy, classpred))
print(arr)

xyxy2 = results2[0].boxes.xyxy.cpu().numpy()
classpred2= results2[0].boxes.cls.cpu().numpy()
arr2 = np.column_stack((xyxy2, classpred2))
print(arr2)

#counting  both back and front fruits
fruits=0
fruits2=0
for i in arr:
    if i[4]==2:
        fruits+=1

for i in arr2:
    if i[4]==2:
        fruits2+=1

print(fruits,fruits2)


