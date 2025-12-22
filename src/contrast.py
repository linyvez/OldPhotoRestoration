import cv2
from skimage.metrics import structural_similarity as ssim

def contrast_process_image(img):
    if img is None:
        print("Image not found.")
        return
    
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    return img, hist

def apply_clahe(img, orig_hist, clipLimit=4.0, tileGridSize=(8,8)):
    rms = img.std()

    if rms > 60:
        print("Image already has good contrast: CLAHE is not applied.")
        return img, orig_hist

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    corrected = clahe.apply(img)

    score = ssim(img, corrected, data_range=255)
    if score < 0.6:
        print("Image is too damaged after applying CLAHE: skip this step or calibrate parameters.")
        return img, orig_hist

    hist = cv2.calcHist([corrected], [0], None, [256], [0, 256])

    return corrected, hist
