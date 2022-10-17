import numpy as np
import cv2
import os

def align_photos(img1, img2):
    #function to align a photo to a template using akaze descriptors and homography matrix
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)

    #get the
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(desc2, desc1, k=2)  # typo fixed

    # filter the best matches using ratio =0.7
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kpts1[match.trainIdx].pt
        points2[i, :] = kpts2[match.queryIdx].pt

    #aply homography matrix
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width, channels = img1.shape
    image_warped = cv2.warpPerspective(img2, h, (width, height), flags=cv2.INTER_NEAREST)
    image_warped = cv2.cvtColor(image_warped, cv2.COLOR_RGB2BGR)
    return image_warped



def filter_circles(image_path):
#get the contours for each circle in the template image of task1
  image = cv2.imread(image_path)
#apply threshold and find contours
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(img_gray, 60, 130, cv2.THRESH_BINARY)

  contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

  list_contours = []

  for c in contours:
    #filter contours that have width and height bigger than 2
    (x,y,w,h) = cv2.boundingRect(c)
    if w > 200 and h > 200:
      list_contours.append(c)


  list_contours = list_contours[1:]
  return list_contours

def detect_circle(point, list_contours):
#having the ten circles of the first board we can detect if a point is a region
#if a circle contains the point and the bigger circle than previous doesnt contain it
  is_in_contour = -1
  for i, contour in enumerate(list_contours):
    result = cv2.pointPolygonTest(contour, point, False)
    if result >= 0:
      is_in_contour = i

  return is_in_contour+1

#create folders for task1
os.mkdir('Iordachescu_Anca_407')
os.mkdir('Iordachescu_Anca_407/Task1')

path_test_data = 'new_test/Task1/'

#load template data
image_path = 'auxiliary_images/template_task1.jpg'
list_contours = filter_circles(image_path)


template = cv2.imread(image_path)

for i in range(1, 26):
    #read images from dataset
    img = None
    if i < 10:
        img = cv2.imread(path_test_data + '0' + str(i) + '.jpg')
    else:
        img = cv2.imread(path_test_data + str(i) + '.jpg')

    #align photo to template
    aligned_photo = align_photos(template, img)

    #change to hsv and obtain a mask to get the barrel of each arrow
    hsv = cv2.cvtColor(aligned_photo, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([8, 42, 57], np.uint8)
    hsv_high = np.array([31, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)

    #apply different morphological operations to get better the barrel of each row
    ret, gray = cv2.threshold(mask, 127, 255, 0)

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(gray, kernel, iterations=1)


    edges = cv2.Canny(img_dilation, 50, 150)

    #draw lines on the barrel of each arrow using color (255, 255, 0)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=4)

    for line in lines:
        new_line = line[0]
        cv2.line(aligned_photo, (new_line[0], new_line[1]), (new_line[2], new_line[3]), (255, 255, 0), 10)

    #now mask contains only regions of the barrel of each row
    #apply some dilatation to make it even clear
    mask1 = cv2.inRange(aligned_photo, (255, 255, 0), (255, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(mask1, kernel, iterations=1)

    #get contour of each barrel of each row
    contours, hierarchy = cv2.findContours(img_dilation,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    file = None
    if i < 10:
        file = 'Iordachescu_Anca_407/Task1/0' + str(i) + '_predicted.txt'
    else:
        file = 'Iordachescu_Anca_407/Task1/' + str(i) + '_predicted.txt'
    f = open(file, "w")
    #number of contours means the number of barrels detected
    f.write(str(len(contours)) + '\n')
    for contour in contours:
        #due to the fact that i only detected the barrel and i have to estimate the tip of the arrow
        #i calculate the centor of each detected barrel and estimate the line that gets through barrel
        #and using a heuristic idea i estimate the tip on a distance of 130 from the center
        M = cv2.moments(contour)
        centru = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        unghi = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
        point_1 = -130 * np.cos(unghi) + centru[0]
        point_2 = -130 * np.sin(unghi) + centru[1]
        pos = detect_circle((point_1, point_2), list_contours)
        f.write(str(pos) + '\n')

    f.close()
