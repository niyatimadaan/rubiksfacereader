import cv2
import numpy as np
from imutils import contours

original = cv2.imread('images/img3.jpg')
image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
mask = np.zeros(image.shape, dtype=np.uint8)
mask2 = np.zeros(image.shape, dtype=np.uint8)

colors = {
    'gray': ([76, 0, 41], [179, 255, 70]),          #1
    'blue': ([100,150,0], [140,255,255]),           #2
    'yellow': ([20, 100, 100], [30, 255, 255]),     #3
    'orange': ([0, 110, 125], [17, 255, 255]),      #4
    'red': ([170, 70, 50], [180, 255, 255]),        #5
    'green': ([25, 52, 72], [102,255, 255]),        #6
    'white':([0, 0, 0],[0, 0, 255])                 #7
    }

# Color threshold to find the squares
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
for color, (lower, upper) in colors.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)

    color_mask = cv2.merge([color_mask, color_mask, color_mask])
    mask = cv2.bitwise_or(mask, color_mask)

gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# Sort all contours from top-to-bottom
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

# Take each row of 3 and sort from left-to-right
cube_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    row.append(c)
    if i % 3 == 0:  
        (cnts, _) = contours.sort_contours(row, method="left-to-right")
        cube_rows.append(cnts)
        row = []
    
distance=[]

# Draw text
number = 0
for row in cube_rows:
    for c in row:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

        cv2.putText(original, "#{}".format(number + 1), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        number += 1
        distance.append( (x,y,0))
        


colorno=1
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
for color, (lower, upper) in colors.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
    color_mask = cv2.merge([color_mask, color_mask, color_mask])
    mask2 = cv2.bitwise_or(mask2, color_mask)
    gray2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    #_,conts, hierarchy = cv2.findContours(color_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    conts = cv2.findContours(gray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if len(conts) == 2 else conts[1]
    cube_rows2=[]
    row2 = []
    if len(conts)==1:
        x,y,w,h = cv2.boundingRect(c)
        distance.append((x,y,colorno))
    
    elif len(conts)!=1:
        for i, cnt in enumerate(conts):
            ce=cnt
            (x,y,w,h) = cv2.boundingRect(ce)
            cube_rows2.append(conts)
            distance.append((x,y,colorno))

    
    colorno+=1

    cv2.imshow('mask5', color_mask)
    cv2.waitKey(500)


print(distance)
print(len(distance))


N=9
arr = [0]*N


for i in range(0,9):
    for j in range(9,len(distance)):
        if distance[i][0] == distance[j][0] and distance[i][1] == distance[j][1]:
            r=int(i/3)
            col=int(i%3)
            arr[i]= distance[j][2]
            break


print(arr)   
cv2.imshow('mask', mask)
cv2.imwrite('mask.png', mask)
cv2.imshow('original', original)

cv2.waitKey(0)
cv2.destroyAllWindows()