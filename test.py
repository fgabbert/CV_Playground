import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)
# print('forward')
# PressKey(W)
# time.sleep(3)
# ReleaseKey(W)
# print('right')
# PressKey(D)
# time.sleep(3)
# ReleaseKey(D)
car_cascade = cv2.CascadeClassifier('cars.xml')

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,255], 3)
    except:
        pass

def process_img(original_image):

    v = np.median(original_image)
    sigma = 0.33
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #processed_img = cv2.Canny(processed_img, threshold1=70, threshold2=60)
    filtered_img = cv2.bilateralFilter(original_image, 9, 75, 200)
    #filtered_img = cv2.GaussianBlur(processed_img,(5,5),0)
    processed_img = cv2.Canny(filtered_img,lower,upper)
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 100, 200, 20)
    draw_lines(processed_img, lines)
    draw_lines(original_image, lines)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.01, 5)
    for (x,y,w,h) in cars:
        cv2.rectangle(original_image, (x,y), (x+w, y+h), (255,0,0), 2)

    return original_image




def main():
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,40,1280,800)))
        new_screen = process_img(screen)
        cv2.imshow('window', cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))

        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()