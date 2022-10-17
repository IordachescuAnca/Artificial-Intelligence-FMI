import numpy as np
import cv2
import os
import cvzone as cvz
from skimage.metrics import structural_similarity

#for this task i have created the task3_temples folder that contains a template for each 3 perspectives in the videos
#i have also created the poze folder that contains 3 folders
#t1 -> have 4 masks that respresents the red circle, the green circle and the two two colored circles for template1
#t2 -> have 4 masks that respresents the red circle, the green circle and the two two colored circles for template2
#t3 -> have 4 masks that respresents the red circle, the green circle and the two two colored circles for template3

def get_contour_template2(point):
    #read the red circle of template2 and check if the point is inside it
    red_zone = cv2.imread('poze/t2/disc rosu.jpg', 0)
    ret, thresh1 = cv2.threshold(red_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh1[point[1], point[0]] == 0:
        return 'b50'
    # read the green circle of template2 and check if the point is inside it
    green_zone = cv2.imread('poze/t2/disc verde.jpg', 0)
    ret, thresh2 = cv2.threshold(green_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh2[point[1], point[0]] == 0:
        return 'b25'

    #define the triangles for each subregion of the darts table
    contours_template2 = [np.array([[180, 0], [160, 153], [661, 84]], dtype=np.int32),
                          np.array([[160, 153], [211, 313], [661, 84]], dtype=np.int32),
                          np.array([[211, 313], [306, 454], [661, 84]], dtype=np.int32),
                          np.array([[306, 454], [441, 558], [661, 84]], dtype=np.int32),
                          np.array([[441, 558], [603, 609], [661, 84]], dtype=np.int32),
                          np.array([[603, 609], [765, 606], [661, 84]], dtype=np.int32),
                          np.array([[765, 606], [913, 564], [661, 84]], dtype=np.int32),
                          np.array([[913, 564], [1034, 460], [661, 84]], dtype=np.int32),
                          np.array([[1034, 460], [1119, 331], [661, 84]], dtype=np.int32),
                          np.array([[1119, 331], [1161, 171], [661, 84]], dtype=np.int32),
                          np.array([[1161, 171], [1153, 11], [661, 84]], dtype=np.int32)]

    #check within which subregion the point is
    contours_indx_template2 = [11, 8, 16, 7, 19, 3, 3, 3,15, 10, 9]
    position = -1
    for i, cnt in enumerate(contours_template2):
        result = cv2.pointPolygonTest(cnt, point, False)
        if result >= 0.0:
            position = contours_indx_template2[i]
            break

    # read the triple ring of template2 and check if the point is inside it
    disk1 = cv2.imread('poze/t2/disc mic 2culori.jpg', 0)
    ret, thresh3 = cv2.threshold(disk1, 127, 255, cv2.THRESH_BINARY)
    if thresh3[point[1], point[0]] == 0:
        return 't' + str(position)

    # read the double ring of template2 and check if the point is inside it
    disk2 = cv2.imread('poze/t2/disc mare 2culori.jpg', 0)
    ret, thresh4 = cv2.threshold(disk2, 127, 255, cv2.THRESH_BINARY)
    if thresh4[point[1], point[0]] == 0:
        return 'd' + str(position)

    #return single ringle + position
    return 's' + str(position)



def get_contour_template3(point):
    # read the red circle of template3 and check if the point is inside it
    red_zone = cv2.imread('poze/t3/disc rosu.jpg', 0)
    ret, thresh1 = cv2.threshold(red_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh1[point[1], point[0]] == 0:
        return 'b50'

    # read the green circle of template3 and check if the point is inside it
    green_zone = cv2.imread('poze/t3/disc verde.jpg', 0)
    ret, thresh2 = cv2.threshold(green_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh2[point[1], point[0]] == 0:
        return 'b25'

    # define the triangles for each subregion of the darts table
    contours_template3 = [np.array([[147, 718], [156   ,555], [653, 644]], dtype=np.int32),
                          np.array([[156  ,555], [201   ,402], [653, 644]], dtype=np.int32),
                          np.array([[201   ,402], [286   ,272], [653, 644]], dtype=np.int32),
                          np.array([[286   ,272], [410   ,180], [653, 644]], dtype=np.int32),
                          np.array([[410   ,180], [553   ,135], [653, 644]], dtype=np.int32),
                          np.array([[553   ,135], [709   ,139], [653, 644]], dtype=np.int32),
                          np.array([[709   ,139], [863   ,186], [653, 644]], dtype=np.int32),
                          np.array([[863   ,186], [997   ,280], [653, 644]], dtype=np.int32),
                          np.array([[997   ,280], [1099   ,410], [653, 644]], dtype=np.int32),
                          np.array([[1099   ,410], [1157   ,564], [653, 644]], dtype=np.int32)]


    contours_indx_template3 = [11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
    #check within which subregion the point is
    position = -1
    for i, cnt in enumerate(contours_template3):
        result = cv2.pointPolygonTest(cnt, point, False)
        if result >= 0.0:
            position = contours_indx_template3[i]
            break
    # read the triple ring of template3 and check if the point is inside it
    disk1 = cv2.imread('poze/t3/disc mic 2culori.jpg', 0)
    ret, thresh3 = cv2.threshold(disk1, 127, 255, cv2.THRESH_BINARY)
    if thresh3[point[1], point[0]] == 0:
        return 't' + str(position)

    # read the double ring of template3 and check if the point is inside it
    disk2 = cv2.imread('poze/t3/disc mare 2culori.jpg', 0)
    ret, thresh4 = cv2.threshold(disk2, 127, 255, cv2.THRESH_BINARY)
    if thresh4[point[1], point[0]] == 0:
        return 'd' + str(position)

    #return single ring + position
    return 's' + str(position)



def get_contour_template1(point):
    # read the red circle of template1 and check if the point is inside it
    red_zone = cv2.imread('poze/t1/disc rosu.jpg', 0)
    ret, thresh1 = cv2.threshold(red_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh1[point[1], point[0]] == 0:
        return 'b50'

    # read the green circle of template1 and check if the point is inside it
    green_zone = cv2.imread('poze/t1/disc verde.jpg', 0)
    ret, thresh2 = cv2.threshold(green_zone, 127, 255, cv2.THRESH_BINARY)

    if thresh2[point[1], point[0]] == 0:
        return 'b25'

    # define the triangles for each subregion of the darts table
    contours_template1 = [np.array([[3, 190], [146, 140], [245, 637]], dtype=np.int32),
                          np.array([[146, 140], [295, 150], [245, 637]], dtype=np.int32),
                          np.array([[295, 150], [447, 198], [245, 637]], dtype=np.int32),
                          np.array([[447, 198], [582, 290], [245, 637]], dtype=np.int32)]

    contours_indx_template1 = [5, 20, 1, 18]

    #check within which subregion the point is
    position = -1
    for i, cnt in enumerate(contours_template1):
        result = cv2.pointPolygonTest(cnt, point, False)
        if result >= 0.0:
            position = contours_indx_template1[i]
            break

    # read the triple ring of template1 and check if the point is inside it
    disk1 = cv2.imread('poze/t1/disc mic 2culori.jpg', 0)
    ret, thresh3 = cv2.threshold(disk1, 127, 255, cv2.THRESH_BINARY)
    if thresh3[point[1], point[0]] == 0:
        return 't' + str(position)


    # read the double ring of template1 and check if the point is inside it
    disk2 = cv2.imread('poze/t1/disc mare 2culori.jpg', 0)
    ret, thresh4 = cv2.threshold(disk2, 127, 255, cv2.THRESH_BINARY)
    if thresh4[point[1], point[0]] == 0:
        return 'd' + str(position)

    #return single ring + position
    return 's' + str(position)



def get_frames(video_path):
    #from a video create a list with all frames
    images = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            images.append(frame)
        else:
            break
    cap.release()
    return images

def get_template(img, t1, t2, t3):
    #function that check a video on which plane is it
    img = img[:, :600]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = t1[:, :600]
    t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
    t2 = t2[:, :600]
    t2 = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY)
    t3 = t3[:, :600]
    t3 = cv2.cvtColor(t3, cv2.COLOR_BGR2GRAY)

    #compute the structural similarity score for each template and photo and return the template that has the higher score
    score1, diff1 = structural_similarity(t1, img, full=True)
    score2, diff2 = structural_similarity(t2, img, full=True)
    score3, diff3 = structural_similarity(t3, img, full=True)
    return np.argmax(np.array([score1, score2, score3])) + 1


#create task3 folder
os.mkdir('Iordachescu_Anca_407/Task3')

path_test_data = 'new_test/Task3/'

#read templates
template1 = cv2.imread('task3_templates/t1.jpg')
template2 = cv2.imread('task3_templates/t2.jpg')
template3 = cv2.imread('task3_templates/t3.jpg')

for i in range(1, 26):
    #get list of frame for each video
    frames = None
    if i < 10:
        frames = get_frames(path_test_data + '0' + str(i) + '.mp4')
    else:
        frames = get_frames(path_test_data + str(i) + '.mp4')

    #get first and last frame of the video
    first_frame = frames[0]
    last_frame = frames[-1]

    diff = cv2.absdiff(first_frame, last_frame)
    #get the template of each video
    template_ind = get_template(first_frame, template1, template2, template3)
    if template_ind == 1:
        first_frame = first_frame[:, :600]
        last_frame = last_frame[:, :600]

    #get the absolute difference between first and last frame
    diff = cv2.absdiff(first_frame, last_frame)

    #apply threshold and get mask
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    (thresh, mask) = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #get contours on the threshold mask
    board_contours, contourFound = cvz.findContours(diff, mask, 70)

    points = []
    #for each contour detected in the photo get the leftest bottom point and add to points list
    for cnt in contourFound:
        box = cnt['bbox']
        points.append((box[0], box[1]+box[3]))
    coordinates = contourFound[len(contourFound) - 1]['bbox']
    print("*****")

    #sort list in order to get the tip of the arrow that would be the lowest point and leftest
    points = sorted(points, key=lambda element: (-element[1], element[0]))
    cv2.circle(board_contours, (points[0][0], points[0][1]), 5, (0, 255, 0), -1)

    #having pos[0] as the tip of the arrow, get the position of each arrow using contour template functions
    pos = None
    if template_ind == 1:
        pos = get_contour_template1((points[0][0], points[0][1]))
    elif template_ind ==  2:
        pos = get_contour_template2((points[0][0], points[0][1]))
    elif template_ind == 3:
        pos = get_contour_template3((points[0][0], points[0][1]))

    file = None
    if i < 10:
        file = 'Iordachescu_Anca_407/Task3/0' + str(i) + '_predicted.txt'
    else:
        file = 'Iordachescu_Anca_407/Task3/' + str(i) + '_predicted.txt'
    f = open(file, "w")
    f.write(pos)
    f.close()



