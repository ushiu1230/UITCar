from math import sqrt
import numpy as np
import cv2
import math
from simple_pid import PID

center_fit_last = np.array([0,0,0])

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
    [x*(0.21), (0.7)*y],
    [x*(0.79), (0.7)*y],
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
    if np.sum(left_fit) == np.sum(right_fit):
        if np.sum(left_fit) >= 300:
            left_fit = np.array([])
        if np.sum(right_fit) <= 300:
            right_fit = np.array([])
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

def find_center_line_for_missing_one_line(image,left_fit,right_fit):
    global center_fit_last
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    point_in_lane = 150,300
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
        print("missing right line")
        #left avaiable
        left_fitx = get_val(ploty,avaiable_fit)
        # max image.shape[1]*0.25+1, min image.shape[1]-image.shape[1]*0.3-1
        center_x = np.clip(left_fitx+100,image.shape[1]*0.25+1,image.shape[1]-image.shape[1]*0.25-1)
        left_fit = avaiable_fit
        right_fit = np.array([])
    else:
        print("missing left line")
        #right avaiable
        right_fitx = get_val(ploty,avaiable_fit)
        center_x = np.clip(right_fitx-100,image.shape[1]*0.25+1,image.shape[1]-image.shape[1]*0.25-1)
        right_fit = avaiable_fit
        left_fit = np.array([])
    center_fit = np.polyfit(ploty, center_x, 2)
    if abs(np.sum(center_fit_last) - np.sum(center_fit)) < 30 or np.sum(center_fit_last) == 0:
        center_fit = np.polyfit(ploty, center_x, 2)
    else:
        center_fit = center_fit_last        
    return center_fit, left_fit, right_fit

def find_center_line_and_update_fit(image,left_fit,right_fit):
    global center_fit_last
    print('center_last', np.sum(center_fit_last))
    left_fit, right_fit = fix_laneline(left_fit, right_fit, np.sum(center_fit_last))
    if np.sum(left_fit) == 0  and np.sum(right_fit) == 0: # missing 2 line:
        center_fit =  np.array([0,0,image.shape[1]/2])
        left_fit_update = np.array([])
        right_fit_update = np.array([])
        center_fit_last = center_fit
        return center_fit, left_fit_update, right_fit_update
    if np.sum(left_fit) == 0 or np.sum(right_fit) == 0: #missing 1 line
        center_fit, left_fit_update, right_fit_update = find_center_line_for_missing_one_line(image,left_fit,right_fit)
        center_fit_last = center_fit
        return center_fit, left_fit_update, right_fit_update
    # none missing line
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    leftx = get_val(ploty, left_fit)
    rightx = get_val(ploty, right_fit)
    center_x = (leftx+rightx)/2
    center_fit = np.polyfit(ploty, center_x, 2)
    if abs(np.sum(center_fit_last) - np.sum(center_fit)) < 30 or np.sum(center_fit_last) == 0:
        center_fit_last = center_fit
    else:
        if int(np.sum(left_fit)) < int(np.sum(right_fit)) and len(left_fit) != 0:
            center_fit_last = center_fit
        else:
            center_fit = center_fit_last
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
    roi = int(center_line.shape[0]*(7/9))+10
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

def errorAngle(center_line,left_fit, right_fit):
    carPosx , carPosy = 300, 150
    angle = 0
    dstx, dsty = find_point_center(center_line)
    left_fit, right_fit = fix_laneline(left_fit, right_fit, dstx)
    leftlane = (np.sum(left_fit))
    rightlane = (np.sum(right_fit))
    #print(leftlane, type(rightlane))
    #leftlane, rightlane = fix_laneline(float(leftlane), float(rightlane))
    dx = dstx - carPosx
    pi = math.acos(-1.0)
    dy = carPosy - dsty
    #print('toa do center lane',dstx, dsty)
    #print('left: ', leftlane,'\tright: ' , rightlane)
    centerlane = (leftlane + rightlane)/2
    centerlane = (300-centerlane)
    #print('center\t', centerlane)
    if leftlane == rightlane and rightlane != 0:
        if leftlane <= 300:
            angle = 10
        else:
            angle = -10
    elif leftlane < rightlane:
        if centerlane >= 0:
            if abs(centerlane) <= 5:
                angle = 0 
            if abs(centerlane) <= 20:
                angle = -sqrt(abs(centerlane))
            elif abs(centerlane) <= 50:
                angle = -abs(centerlane)/2
                if angle < -30:
                    angle = (math.atan(dx / dy) * 180 / pi)/2.8
            else:
                angle = (math.atan(dx / dy) * 180 / pi)/2.8
        else:
            if abs(centerlane) <= 5:
                angle = 0
            if abs(centerlane) <= 20:
                angle = (sqrt(abs(centerlane)))
            elif abs(centerlane) <= 50:
                angle = (abs(centerlane)/2)
                if angle > 30:
                    angle = (math.atan(dx / dy) * 180 / pi)/2.8
            else:
                angle = (math.atan(dx / dy) * 180 / pi)/2.8
    else:
        if abs(centerlane) <= 20:
            angle = -(sqrt(abs(centerlane)))
        elif abs(centerlane) <= 50:
            angle = -(abs(centerlane)/2)
        else:
            angle = (math.atan(dx / dy) * 180 / pi)/2.8
    #print('angle: ',angle)
    return angle

def calcul_speed(steer_angle):
    max_speed = 20
    max_angle = 30
    if (steer_angle >= 1 and steer_angle < 4)  or  (steer_angle > -4 and  steer_angle <= -1):
        max_speed*=0.8
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle 
    
    elif (steer_angle >= 4 and steer_angle < 8) or  (steer_angle > -8 and  steer_angle <= -4):
        max_speed*=0.9
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle

    elif (steer_angle >= 8 and steer_angle < 12) or  (steer_angle > -12 and  steer_angle <= -8):
        max_speed*=0.7
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle


    elif (steer_angle >= 12 and steer_angle < 17) or  (steer_angle > -17 and  steer_angle <= -12):
        max_speed*=0.3
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle

    
    elif (steer_angle >= 17 and steer_angle < 26) or  (steer_angle > -26 and  steer_angle <= -17):
        max_speed*=0.2
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle

    elif abs(steer_angle >= 26):
        max_speed*=0.2
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle
    return max_speed


################## find line avaiable ######################
# def line_processing(image):
#    binary_image =  binary_pipeline(image)
#    bird_view, inverse_perspective_transform =  warp_image(binary_image)
#    left_fit, right_fit = track_lanes_initialize(bird_view)
#    return left_fit, right_fit,bird_view, inverse_perspective_transform
################## Draw lane avaiable #######################
# def draw_lane(image, bird_view, left_fit, right_fit, inverse_perspective_transform):
#     left_fit, right_fit = check_fit_duplication(left_fit,right_fit)
#     center_fit, left_fit, right_fit = find_center_line_and_update_fit(image,left_fit,right_fit) # update left, right line
#     colored_lane, center_line = lane_fill_poly(bird_view,image,center_fit,left_fit,right_fit, inverse_perspective_transform)
#     cv2.imshow("lane",colored_lane)
#     return center_line


kp = 0.7
ki = 0
kd = 0.1
pid = PID(kp,ki,kd, setpoint= 0)

pid.output_limits = (-30,30)

def get_speed_angle(center_line,left_fit, right_fit): 
# calculate speed and angle
   steer_angle =  errorAngle(center_line,left_fit, right_fit)
   steer_angle_loss = pid(-(steer_angle*0.3))
   print(steer_angle_loss)
   speed_current = calcul_speed(steer_angle_loss)
   return speed_current, steer_angle
