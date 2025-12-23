import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

LOWER_THERSHOLD = 0
UPPER_THERSHOLD = 100


def sorted_points(points):
    center = [0, 0]
    for point in points:
        center[0] += point[0]
        center[1] += point[1]

    center[0] /= len(points)
    center[1] /= len(points)

    top_right = None
    top_left = None

    bottom_right = None
    bottom_left = None

    for point in points:
        if point[0] > center[0] and point[1] > center[1]:
            bottom_right = point
        elif point[0] < center[0] and point[1] > center[1]:
            bottom_left = point
        elif point[0] > center[0] and point[1] < center[1]:
            top_right = point
        elif point[0] < center[0] and point[1] < center[1]:
            top_left = point
        else:
            assert False

    return [top_left, top_right, bottom_right, bottom_left]


def get_best_candidate(img):
    img = cv.Canny(img, LOWER_THERSHOLD, UPPER_THERSHOLD)
    img = cv.dilate(img, np.ones((3, 3), np.uint8))

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    long_contours = [contour for contour in contours if len(contour) > 500]

    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width

    candidates = []
    for cnt in long_contours:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

        points = [approx[0][0], approx[1][0], approx[2][0], approx[3][0]]
        ordered_points = sorted_points(points)

        if len(approx) != 4:
            continue

        for point in ordered_points:
            if point is None:
                break
        
        if point is None:
            continue

        cnt_height = abs(ordered_points[0][1] - ordered_points[2][1])
        cnt_width = abs(ordered_points[0][0] - ordered_points[1][0])

        expected_area = cnt_height * cnt_width
        calculated_area = cv.contourArea(cnt)

        if calculated_area > 0.6 * img_area and 0.8 * expected_area < calculated_area and calculated_area < 1.2 * expected_area:
            candidates.append((ordered_points, calculated_area))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[1])
    candidate = candidates[0]

    candidate = np.array([points for points in candidate[0]]).astype(np.float32)
    return candidate


def preprocess_image(img):

    img_height, img_width = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    candidate = get_best_candidate(gray)

    if candidate is None:
        print("No corners detected. Skipping perspective correction.")
        return img
    
    dst = np.array([(0, 0), (img_width-1, 0), (img_width-1, img_height-1), (0, img_height-1)]).astype(np.float32)

    matrix = cv.getPerspectiveTransform(candidate, dst)
    result = cv.warpPerspective(img, matrix, (img_width, img_height))

    return result


if __name__ == '__main__':
 
    filename = '../photos/1.png'
    img_raw = cv.imread(filename)
    # gray = cv.cvtColor(img_raw,cv.COLOR_BGR2GRAY)
    
    # img = gray
    result = preprocess_image(img_raw)

    # cv.drawContours(img_raw, long_contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
