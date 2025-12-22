import cv2
import numpy as np

def remove_scratches(img_rgb, threshold=35, radius=3):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    _, mask = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            radius_eye = int(ew / 2.2)
            cv2.circle(roi_mask, eye_center, radius_eye, 0, -1)

    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    
    restored = cv2.inpaint(img_rgb, mask, radius, cv2.INPAINT_NS)

    return restored, mask