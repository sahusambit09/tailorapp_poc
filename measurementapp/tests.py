import cv2 as cv
import numpy as np
webcam = False
cThr = [100, 100]
cap = cv.VideoCapture(0)
cap.set(50, 160)
cap.set(3, 1920)
cap.set(4, 1080)
img = cv.imread('human_900x1200.jpeg')
imgBlur = cv.GaussianBlur(img, (5, 5), 1)
imgray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
imgCanny = cv.Canny(imgray, cThr[0], cThr[1])
ret, threshold = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print(len(str(contours)))
if webcam:
    success, img = cap.read()
cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.imshow('Image', img)
cv.imshow('BlUR', imgBlur)
cv.imshow('Gray', imgray)
cv.imshow('Canny', imgCanny)
cv.waitKey(0)

# while True:

# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(imgray)
