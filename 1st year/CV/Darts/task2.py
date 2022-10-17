import numpy as np
import cv2
import os


def sliding_window(image, stepSize, windowSize):
    #returns all windows from a image having windowSize length and the step of the sliding window is stepsize
    images = []
    for y in range(0,image.shape[0] - windowSize, stepSize):
        for x in range(0,image.shape[1] - windowSize, stepSize):
            img = image[y:(y+windowSize), x:(x+windowSize)]
            images.append((y, x, img))

    return images

#i have also created the poze/template_task2 folder that has 4 masks that respresents the red circle, the green circle and the two two colored circles for template


def get_contour_template1(point):
    #read the red circle of template and check if the point is inside it
    red_zone = cv2.imread('poze/template_task2/disc rosu.jpg', 0)
    ret, thresh1 = cv2.threshold(red_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh1[point[1], point[0]] == 0:
        return 'b50'

    #read the green circle of template and check if the point is inside it
    green_zone = cv2.imread('poze/template_task2/disc verde.jpg', 0)
    ret, thresh2 = cv2.threshold(green_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh2[point[1], point[0]] == 0:
        return 'b25'

    # define the triangles for each subregion of the darts table
    contours_template = [np.array([[945, 1034], [1163, 941], [1326, 1853]], dtype=np.int32),
                     np.array([[1163, 941], [1384, 950], [1326, 1853]], dtype=np.int32),
                     np.array([[1384, 950], [1582, 1043], [1326, 1853]], dtype=np.int32),
                     np.array([[1582, 1043], [1748, 1201], [1326, 1853]], dtype=np.int32),
                     np.array([[1748, 1201], [1861,   1406], [1326, 1853]], dtype=np.int32),
                     np.array([[1861,   1406], [1926,   1634], [1326, 1853]], dtype=np.int32),
                     np.array([[1926,   1634], [1928,   1890], [1326, 1853]], dtype=np.int32),
                     np.array([[1928,   1890], [1892,   2142], [1326, 1853]], dtype=np.int32),
                    np.array([[1892,   2142], [1798,   2355], [1326, 1853]], dtype=np.int32),
                    np.array([[1798,   2355], [1652,   2538], [1326, 1853]], dtype=np.int32),
                    np.array([[1652,   2538], [1478,   2687], [1326, 1853]], dtype=np.int32),
                    np.array([[1478,   2687], [1264 ,  2766], [1326, 1853]], dtype=np.int32),
                    np.array([[1264 ,  2766], [1042,   2738], [1326, 1853]], dtype=np.int32),
                    np.array([[1042,   2738], [830,   2603], [1326, 1853]], dtype=np.int32),
                    np.array([[830,   2603], [650,   2393], [1326, 1853]], dtype=np.int32),
                    np.array([[650,   2393], [542,   2100], [1326, 1853]], dtype=np.int32),
                    np.array([[542,   2100], [524,   1779], [1326, 1853]], dtype=np.int32),
                    np.array([[524,   1779], [590,   1467], [1326, 1853]], dtype=np.int32),
                    np.array([[590,   1467], [731,   1192], [1326, 1853]], dtype=np.int32),
                    np.array([[731,   1192], [945, 1034], [1326, 1853]], dtype=np.int32)
                     ]

    positions_contours = [5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12]

    # check within which subregion the point is
    position = -1
    for i, cnt in enumerate(contours_template):
        result = cv2.pointPolygonTest(cnt, point, False)
        if result >= 0.0:
            position = positions_contours[i]
            break

    # read the triple ring of template and check if the point is inside it
    disk1 = cv2.imread('poze/template_task2/disc mic 2culori.jpg', 0)
    ret, thresh3 = cv2.threshold(disk1, 127, 255, cv2.THRESH_BINARY)
    if thresh3[point[1], point[0]] == 0:
        return 't' + str(position)

    # read the double ring of template and check if the point is inside it
    disk2 = cv2.imread('poze/template_task2/disc mare 2culori.jpg', 0)
    ret, thresh4 = cv2.threshold(disk2, 127, 255, cv2.THRESH_BINARY)
    if thresh4[point[1], point[0]] == 0:
        return 'd' + str(position)

    #return single region + position
    return 's' + str(position)


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

def iou(corner1, corner2, step):

    #function that computes intersection over union score
    #it is going to be used for sliding window to remove the windows that overlaps a lot with others
    corner1_up = corner1[0]
    corner1_left = corner1[1]
    corner1_down = corner1_up + step
    corner1_right = corner1_left + step

    corner2_up = corner2[0]
    corner2_left = corner2[1]
    corner2_down = corner2_up + step
    corner2_right = corner2_left + step

    int_up = max(corner1_up, corner2_up)
    int_left = max(corner1_left, corner2_left)
    int_down = min(corner1_down, corner2_down)
    int_right = min(corner1_right, corner2_right)

    interArea = max(0, int_down - int_up + 1) * max(0, int_right - int_left + 1)
    res = interArea/(step*step + step*step - interArea)
    #print(res, interArea, int_down - int_up, int_right - int_left)
    return res


#create task2 folder
os.mkdir('Iordachescu_Anca_407/Task2')

path_test_data = 'new_test/Task2/'

#read template
template = cv2.imread('auxiliary_images/template_task2.jpg')

#step of sliding window
step_sw = 200

for i in range(1, 26):
    #read each image from the dataset
    img = None
    if i < 10:
        img = cv2.imread(path_test_data + '0' + str(i) + '.jpg')
    else:
        img = cv2.imread(path_test_data + str(i) + '.jpg')


    #align the photo to the template
    aligned_photo = align_photos(template, img)

    #compute difference to the template and aligned photo
    diff = cv2.absdiff(template, aligned_photo)

    #create a mask to recognize better the arrows
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray[gray < 50] = 0
    gray[gray >= 50] = 255

    #create windows using sliding window
    windows = sliding_window(gray, 10, step_sw)

    new_windows = []
    for (y, x, image) in windows:
        #only take into consideration the windows that has a ratio of white pixels bigger than 0.25
        #this method will get the number of arrows
        unique, counts = np.unique(image, return_counts=True)
        frequency = np.asarray((unique, counts)).T
        if frequency.shape[0] == 2:
            ratio_white = frequency[1][1] / (step_sw * step_sw)
            if ratio_white > 0.25:
                new_windows.append((x, y, ratio_white))

    #sort the filtered windows by the ratio of white pixels
    new_windows = sorted(new_windows, key=lambda x: x[2], reverse=True)

    #remove the overlapping windows using intersection over union
    filtered_windows = []
    for (x1, y1, r1) in new_windows:
        add = True
        for (x2, y2) in filtered_windows:
            ratio = iou((x1, y1), (x2, y2), step_sw)
            if ratio > 0.1:
                add = False

        if add:
            filtered_windows.append((x1, y1))


    #got the number of arrows
    number_arrows = len(filtered_windows)

    #detect positions
    aligned_photo = align_photos(template, img)
    #change to hsv and obtain a mask to get the barrel of each arrow
    hsv = cv2.cvtColor(aligned_photo, cv2.COLOR_BGR2HSV)
    hsv_low = np.array([8, 42, 57], np.uint8)
    hsv_high = np.array([31, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)

    #apply different morphological operations to get better the barrel of each row
    ret, gray = cv2.threshold(mask, 127, 255, 0)
    gray2 = gray.copy()
    mask = np.zeros(gray.shape, np.uint8)


    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(gray, kernel, iterations=1)


    edges = cv2.Canny(img_dilation, 50, 150)

    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=4)

    #draw lines on the barrel of each arrow using color (255, 255, 0)
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

    # Display results

    #select contours that have the biggest number_arrows area
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    contours = cntsSorted[-number_arrows:]

    if i < 10:
        file = 'Iordachescu_Anca_407/Task2/0' + str(i) + '_predicted.txt'
    else:
        file = 'Iordachescu_Anca_407/Task2/' + str(i) + '_predicted.txt'
    f = open(file, "w")
    # number of contours means the number of barrels detected
    f.write(str(len(contours)) + '\n')

    for contour in contours:
        #due to the fact that i only detected the barrel and i have to estimate the tip of the arrow
        #i calculate the centor of each detected barrel and estimate the line that gets through barrel
        #and using a heuristic idea i estimate the tip on a distance of 150 from the center
        M = cv2.moments(contour)
        centru = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        unghi = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
        point_1 = -150 * np.cos(unghi) + centru[0]
        point_2 = -150 * np.sin(unghi) + centru[1]
        pos = get_contour_template1((int(point_1), int(point_2)))
        f.write(pos + '\n')
    f.close()
    print(i)
