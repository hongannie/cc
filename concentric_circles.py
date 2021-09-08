#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:37:24 2020

@author: annie
"""

import cv2
import numpy as np
import glob
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
pathname_list = glob.glob("*.png")

#empty lists
values = []
radii= []
whites = []     # of white pixels
covper = []     # % coverage values
path = []       # names of all the files
covcc = []      # of white pixels in each concentric circle

for pathname in pathname_list:
    # read image
    img = cv2.imread(pathname)
    path.append(pathname)
    #plt.imshow(img)

    
    #create a mask
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # convert to HSV and extract saturation channel, threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #_, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #gray = cv2.medianBlur(gray,5)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    #plt.imshow(thresh)

    """
    uncomment in order to show the thresholded image
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    kernel = np.ones((10,10),np.uint8)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)
    opening2 = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel, iterations = 1)
    _, contours, hierarchy = cv2.findContours(opening2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    radius = (img.shape[0])/2
    #print(extRight[0], extLeft[0])
    #print(radius)
     
    #find center of image
    M = cv2.moments(cnt)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    cx = int(img.shape[1]/2)
    cy = int(img.shape[0]/2)
    print(cx, cy)
    
    #empty list of white pixels
    whitenum = []
   
    for i in range(1,100):
        if i*30<radius:
            #print("radius",i*30)
            radii.append(i*30)
            cv2.circle(mask,(cx,cy), i*30, 255, -1)
            #res = cv2.bitwise_and(thresh, thresh, mask=mask)
            res = cv2.bitwise_and(gray, gray, mask=mask)
            #white = np.sum(res == 255)
            white = np.sum(res >= 50)
            coverage = white/(pi*((i*30)**2))
            covper.append(coverage)
            whites.append(white)
            whitenum.append(white)
            values.append((i*30, coverage))
            path.append(' ')
            
            """
            #UNCOMMENT IF YOU WANT TO SHOW IMAGES
            cv2.putText(res,'Pixel count: '+str(white),(30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('img', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i in range(5):    # extra wait keys, destroywindow not working
                cv2.waitKey(1)
            """
        else:
            #res = cv2.bitwise_and(thresh, thresh, mask=opening)
            res = cv2.bitwise_and(gray, gray, mask=opening2)
            white = np.sum(res >= 50)
            cv2.putText(img,'white: '+str(white),(30,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('img', res)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            for i in range(5):    # extra wait keys, destroywindow not working
                cv2.waitKey(1)
            break
        
    #end path
    path.pop()
    
    #coverage in each donut
    i = 1
    covcc.append(whitenum[0]/(pi*((1*30)**2)))
    while i < len(whitenum):
        diff = whitenum[i] - whitenum[i-1]
        donutarea = pi*(((i+1)*30)**2) - pi*((i*30)**2)
        covdonut = diff/donutarea
        covcc.append(covdonut)
        i += 1
    
    
#write to a file
file = open("cc.txt", "a+") 
for x in values:
    file.write('{}\n'.format(str(x)))
file.close()

print(covcc)
# create pandas dataframe
# create 2 columsn with radius and coverage values
df = pd.DataFrame({'file': path,
                   'radius': radii,
                  '% coverage': covper,
                  'donut % cov': covcc})
writer = pd.ExcelWriter('cc.xlsx', engine='xlsxwriter')
# sheet name
df.to_excel(writer, sheet_name='genotype')
writer.save()  

