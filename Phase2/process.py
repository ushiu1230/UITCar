from math import sqrt
import numpy as np
import cv2
import math
import statistics
import time
from signboard_detect import *

center_fit_last = np.array([0,0,0])
global_left_fit = np.array([0,0,0])
global_right_fit = np.array([0,0,0])

speed_list = [0]*10
ang_list = [0]*10
time_list = [0]*10
turn = 0
start_time = time.time()

def stdev_list(list, point):
    list.append(point)
    del list[:-10]
    #print(list)
    avg = sum(list)/len(list)
    avg += statistics.stdev(list)
    return avg

def rgb_select(img, thresh=(0, 255)):
    R = img[:,:,2] 
    G = img[:,:,1]
    B = img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R >= thresh[0]) & (R <= thresh[1]) & (G >= thresh[0]) & (G <= thresh[1]) & (B >= thresh[0]) & (B <= thresh[1])] = 1
    return binary_output

def line_in_shadow(img, thresh1=(0,255),thresh2=(0,255),thresh3=(0,255)):
    R = img[:,:,2] 
    G = img[:,:,1]
    B = img[:,:,0]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(R)
    binary_output[(R >= thresh1[0]) & (R <= thresh1[1]) & (G >= thresh2[0]) & (G <= thresh2[1]) & (B >= thresh3[0]) & (B <= thresh3[1])] = 1
    return binary_output

def binary_pipeline(img):
    img_copy = cv2.GaussianBlur(img, (3, 3), 0)
    binary = cv2.Canny(img_copy, threshold1= 100, threshold2=200)
    return binary


def hsv_select(img, lower=np.array([10, 0, 0]), upper =np.array([180, 50,210])):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)
    # cv2.imshow("mask", hsv_img)
    return mask

def lane_in_shadow(img, lower=np.array([45, 55, 60]), upper =np.array([55, 70,80])):
    R = img[:,:,2] 
    G = img[:,:,1]
    B = img[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R >= lower[0]) & (R <= upper[0]) & (G >= lower[1]) & (G <= upper[1]) & (B >= lower[2]) & (B <= upper[2])] = 255
    # cv2.imshow("hsv_img", img)
    return binary_output

def warp_image(img):
    
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]

    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    ## my source
    source_points = np.float32([
    [0, y],
    [x*(0.17), (0.77)*y],
    [x*(0.83), (0.77)*y],
    [x, y]
    ])
    
    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    
    warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)

    return warped_img, inverse_perspective_transform

def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

def check_lane_inds(left_lane_inds, right_lane_inds):
    countleft = 0
    countright = 0
    missing_one_line = False
    for x in range(9):
        left = np.asarray(left_lane_inds[x])
        right = np.asarray(right_lane_inds[x])
        if len(left) == 0:
            countleft+=1
        if len(right) == 0:
            countright+=1
        if len(left) == len(right) and len(left) !=0 and len(right) != 0:
            if (left == right).all():
                missing_one_line = True
    if missing_one_line:
        if countleft == countright:
            return left_lane_inds, right_lane_inds
        if countleft < countright:
            return left_lane_inds, []
        return [], right_lane_inds
    if countleft >= 6:
        return [], right_lane_inds
    if countright >= 6:
        return left_lane_inds, []
    return left_lane_inds,right_lane_inds

def track_lanes_initialize(binary_warped):   
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint+100])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint+100
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 25
    # Set minimum number of pixels found to recenter window
    minpix = 15
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # cv2.imshow('out_img',out_img)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    
    left_lane_inds,right_lane_inds = check_lane_inds(left_lane_inds,right_lane_inds)
    if len(left_lane_inds) != 0:
        left_lane_inds = np.concatenate(left_lane_inds)
    if len(right_lane_inds) !=0:
        right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    left_fit = np.array([])
    right_fit = np.array([])
    #print(len(leftx), len(rightx))
    if len(leftx) != 0:
        left_fit  = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit  = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def check_fit_duplication(left_fit, right_fit):
    if len(left_fit) == 0 or len(right_fit) == 0:
        return left_fit, right_fit
    # print(left_fit[2], right_fit[2])
    if abs(left_fit[0] - right_fit[0]) < 0.1:
        if abs(left_fit[1] - right_fit[1]) < 0.4:
            if abs(left_fit[2] - right_fit[2]) < 30:
                return left_fit, []
    return left_fit, right_fit


#### UPDATE #####
def get_point_in_lane(image):
    warp,_ = warp_image(image)
    lane_image = hsv_select(warp)
    lane_shadow = lane_in_shadow(warp)
    lane = cv2.bitwise_or(lane_image,lane_shadow)
    histogram_x = np.sum(lane[:,:], axis=0)
    histogram_y = np.sum(lane[:,:], axis=1)
    lane_x = np.argmax(histogram_x)
    lane_y = np.argmax(histogram_y)
    dst = abs(lane_y-lane.shape[0])
    if dst < 200:
        for y in range(lane_y,0,-1):
            if lane[y][lane_x] == 255:
                return [y, lane_x]
    else:
        for y in range(lane_y,lane_y+dst-1):
            if lane[y][lane_x] == 255:
                return [y, lane_x]
    # if dst == 0
    
    return 0,0

def update_fit_value(center_fit, left_fit, right_fit):
    global center_fit_last
    global global_left_fit
    global global_right_fit

    center_last_sum = np.sum(center_fit_last)
    left_last_sum = np.sum(global_left_fit)
    right_last_sum = np.sum(global_right_fit)

    center_sum = np.sum(center_fit)
    left_fit_sum = np.sum(left_fit)
    right_fit_sum = np.sum(right_fit)

    if left_last_sum == 0 and right_last_sum == 0:
        global_right_fit = right_fit
        global_left_fit = left_fit
    else:
        if left_fit_sum == 0:
            if right_fit_sum == 0:
                return center_fit, left_fit, right_fit
            else:
                center_fit[0] = (center_fit[0] + right_fit[0])
                center_fit[1] = (center_fit[1] + right_fit[1])
                global_right_fit = right_fit
                center_fit_last = center_fit
                #print('offset_right',type(offset_right))
        else:
            if right_fit_sum == 0:
                center_fit[0] = (center_fit[0] + left_fit[0])
                center_fit[1] = (center_fit[1] + left_fit[1])
                global_left_fit = left_fit
                center_fit_last = center_fit
                #print('offset_left',type(offset_left))
            else:
                global_right_fit = right_fit
                global_left_fit = left_fit
                center_fit_last = center_fit

    if abs(center_last_sum - center_sum) > 30 and center_last_sum != 0:
        if left_fit_sum == 0 and right_fit_sum == 0:
            center_fit_last = center_fit
            global_right_fit = right_fit
            global_left_fit = left_fit
            #print('reset center')
        else:
            center_fit = center_fit_last
            #print('not update')
    else:
        center_fit_last = center_fit
        global_right_fit = right_fit
        global_left_fit = left_fit
        #print('update')

    #print(np.sum(right_fit - left_fit))

    return center_fit, left_fit, right_fit

def find_center_line_for_missing_one_line(image,left_fit,right_fit):
    global center_fit_last
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    point_in_lane = 150,305
    #print('point_in_lane:',point_in_lane)
    # cv2.circle(image,(point_in_lane[1],point_in_lane[0]),1,(0,0,255),8)
    avaiable_fit =  left_fit
    center_x = np.array([])
    #print('left lengh',np.sum(left_fit))
    left_fit, right_fit = fix_laneline(left_fit, right_fit,np.sum(center_fit_last))
    #print(np.sum(left_fit), np.sum(right_fit))
    if np.sum(left_fit) == 0:
        avaiable_fit = right_fit
    #print('avaiable_fit ',np.sum(avaiable_fit))
    val = point_in_lane[1] - get_val(point_in_lane[0],avaiable_fit)
    #print('val: ',val)
    if val > 0:
        #print("missing right line")
        #left avaiable
        left_fitx = get_val(ploty,avaiable_fit)
        # max image.shape[1]*0.25+1, min image.shape[1]-image.shape[1]*0.3-1
        center_x = np.clip(left_fitx+100,image.shape[1]*0.25+1,image.shape[1]-image.shape[1]*0.25-1)
        left_fit = avaiable_fit
        right_fit = np.array([])
    else:
        #print("missing left line")
        #right avaiable
        right_fitx = get_val(ploty,avaiable_fit)
        center_x = np.clip(right_fitx-100,image.shape[1]*0.25+1,image.shape[1]-image.shape[1]*0.25-1)
        right_fit = avaiable_fit
        left_fit = np.array([])
    center_fit = np.polyfit(ploty, center_x, 2)
    #center_fit, left_fit, right_fit = update_fit_value(center_fit, left_fit, right_fit)
    return center_fit, left_fit, right_fit

def find_center_line_and_update_fit(image,left_fit,right_fit):
    global center_fit_last
    global global_left_fit
    global global_right_fit
    #print('center_last', np.sum(center_fit_last))
    left_fit, right_fit = fix_laneline(left_fit, right_fit, np.sum(center_fit_last))
    if np.sum(left_fit) == 0  and np.sum(right_fit) == 0: # missing 2 line:
        center_fit =  np.array([0,0,image.shape[1]/2])
        left_fit_update = np.array([])
        right_fit_update = np.array([])
        center_fit, left_fit_update, right_fit_update = update_fit_value(center_fit, left_fit_update, right_fit_update)
        return center_fit, left_fit_update, right_fit_update
    if np.sum(left_fit) == 0 or np.sum(right_fit) == 0: #missing 1 line
        center_fit, left_fit_update, right_fit_update = find_center_line_for_missing_one_line(image,left_fit,right_fit)
        center_fit, left_fit_update, right_fit_update = update_fit_value(center_fit, left_fit_update, right_fit_update)
        return center_fit, left_fit_update, right_fit_update
    # none missing line
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    leftx = get_val(ploty, left_fit)
    rightx = get_val(ploty, right_fit)
    center_x = (leftx+rightx)/2
    center_fit = np.polyfit(ploty, center_x, 2)
    center_fit, left_fit, right_fit = update_fit_value(center_fit, left_fit, right_fit)
    return center_fit, left_fit, right_fit

def lane_fill_poly(binary_warped,undist,center_fit,left_fit,right_fit, inverse_perspective_transform):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if len(left_fit) == 0:
        left_fit = np.array([0,0,1])
    if len(right_fit) == 0:
        right_fit = np.array([0,0,binary_warped.shape[1]-1])
    left_fitx = get_val(ploty,left_fit)
    right_fitx = get_val(ploty,right_fit)
    center_fitx = get_val(ploty,center_fit)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    center_color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast x and y for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
    #print(pts_center)
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane 
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.fillPoly(center_color_warp, np.int_([pts_center]),(0,0,255))
    #cv2.imshow('center_color_warp',center_color_warp)
    # Warp using inverse perspective transform
    newwarp = cv2.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
    center_line = cv2.warpPerspective(center_color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
   
    result = cv2.addWeighted(undist, 1, newwarp, 0.7, 0.3)
    result = cv2.addWeighted(result,1,center_line,0.7,0.3)
    return result, center_line

############################## calcul steer angle #############################
def find_point_center(center_line):
    roi = int(center_line.shape[0]*(0.8))+10
    for y in range(roi,center_line.shape[0]):
        for x in range(center_line.shape[1]):
            if center_line[y][x][2] == 255:
                cv2.circle(center_line,(x,y),1,(255,0,0),7)
                # cv2.imshow('center_point',center_line)
                return x,y
    return 0,0

def fix_laneline(left_fit, right_fit, dstx):
    global center_fit_last
    sum_left = int(np.sum(left_fit))
    sum_right = int(np.sum(right_fit))

    #print('center_fit_last[2]', np.sum(center_fit_last))
    if sum_left == sum_right:
        if sum_left >= dstx:
            left_fit = np.array([])
            right_fit = right_fit
            #print('left_fix: ', left_fit, "right_fix: ",right_fit)
        elif sum_right < dstx:
            left_fit = left_fit
            right_fit = np.array([])
    else:
        if sum_left == 0 and sum_right < dstx:
            left_fit = right_fit
            right_fit = np.array([])
        elif sum_right == 0 and sum_left > dstx:
            right_fit = left_fit
            left_fit = np.array([])
    
    return left_fit, right_fit

def func_turn(turn,state):
    print('turn func')
    if state == 1:
        if turn == 2:
            return -2
        elif turn == 3:
            return 2
        return 0
    elif state == 2:
        return -25
    elif state == 3:
        return 25
    elif state == 4:
        #cam phai
        if turn == 1:
            return -25
        elif turn == 2:
            return 2
    elif state == 3:
        #cam trai
        if turn == 1:
            return 25
        elif turn == 3:
            return 2
    elif state == 6:
        if turn == 2:
            return 25
        elif turn == 3:
            return -25
    print('ko vo state')
    return 0

def errorAngle1(center_line, state):
    global global_left_fit
    global global_right_fit
    global turn
    carPosx , carPosy = 305, 150
    angle = 0
    dstx, dsty = find_point_center(center_line)
    #left_fit, right_fit = fix_laneline(left_fit, right_fit, dstx)
    leftlane = np.sum(global_left_fit)
    rightlane = np.sum(global_right_fit)
    #print(leftlane, rightlane,)
    #leftlane, rightlane = fix_laneline(float(leftlane), float(rightlane))
    dx = dstx - carPosx
    pi = math.acos(-1.0)
    dy = carPosy - dsty
    print('turn:\t',turn)
    print('state:\t',state)
    print('left: ', leftlane,'\tright: ' , rightlane)
    centerlane = (leftlane + rightlane)/2
    centerlane = (carPosx-centerlane)
    #print('center\t', (leftlane + (rightlane-leftlane)/2))
    #print('center', centerlane)
    if turn != 0 and state != 0:
        print('xu ly sign')
        angle = func_turn(turn,state)
        if rightlane != 0 and leftlane != 0:
            turn = 0
            state = 0
        return angle
    if rightlane == 0 and leftlane == 0:
        angle = 0
    elif rightlane != 0 and leftlane != 0:
        if rightlane - leftlane < 590:
            if centerlane > 0:
                if centerlane > 10:
                    if centerlane > 20:
                        if centerlane > 40:
                            angle = -12
                        else:
                            angle = -8
                    else:
                        angle = -5
                else:
                    angle = -3
            else:
                centerlane_abs = abs(centerlane)
                if centerlane_abs > 10:
                    if centerlane_abs > 20:
                        if centerlane_abs > 40:
                            angle = 12
                        else:
                            angle = 8
                    else:
                        angle = 5
                else:
                    angle = 3
        else:
            print('Nga tu hoac ba chu T\n')
            turn = 1
            angle = func_turn(turn,state)
            return angle
    else:
        if leftlane != 0:
            if leftlane > 220:
                print('cua phai')
                if leftlane < 235:
                    if leftlane > 100:
                        angle = 7
                    else:
                        angle = -7
                else:
                    if leftlane < 250:
                        angle = 20
                    else:
                        angle = 25
            elif leftlane >= 145 and leftlane <= 210:
                print('nga ba phai')
                turn = 2
                angle = func_turn(turn,state)
                return angle
        elif rightlane != 0:
            if rightlane <= 505 and rightlane >= 400:
                print('nga ba trai')
                turn = 3
                angle = func_turn(turn,state)
                print(angle)
                return angle
            elif rightlane < 700:
                print('cua trai')
                if rightlane > 475:
                    if rightlane > 100:
                        angle = -7
                    else:
                        angle = 7
                else:
                    if rightlane > 400:
                        angle = -20
                    else:
                        angle = -25
    return angle
    
def errorAngle(center_line,left_fit, right_fit):
    carPosx , carPosy = 305, 150
    dstx, dsty = find_point_center(center_line)
    print('dstx, dsty',dstx, dsty)
    if dstx == carPosx:
        return 0
    if dsty == carPosy:
        if dstx < carPosx:
            return -30
        else:
            return 30
    pi = math.acos(-1.0)
    dx = dstx - carPosx
    dy = carPosy - dsty
    print(dx, dy)

    if dx < 0:
        angle = (math.atan(-dx / dy) * -180 / pi)
        if abs(angle) > 4:
            if abs(angle) > 10:
                if abs(angle) > 18:
                    if abs(angle) > 23:
                        angle = 30
                angle = angle/9
            angle = angle/3
    else:
        angle = (math.atan(dx / dy) * 180 / pi)
        if angle > 4:
            if angle > 10:
                if angle > 18:
                    if angle > 23:
                        angle = 30
                angle = angle/9
            angle = angle/3
    return angle


def calcul_speed(steer_angle):
    max_speed = 22
    abs_steering = abs(steer_angle)
    if abs_steering < 2:
        return max_speed
    elif abs_steering < 8:
        return max_speed*0.9
    elif abs_steering < 15:
        return max_speed*0.7
    elif abs_steering < 20:
        return max_speed*0.5
    elif abs_steering < 40:
        return max_speed*0.3
    else:
        return max_speed*0.1

def get_speed_angle(center_line,state):

    steer_angle =  errorAngle1(center_line, state)
    print('angle: ',steer_angle)
    speed_current = calcul_speed(steer_angle)
    return speed_current, steer_angle
