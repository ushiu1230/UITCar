from turtle import speed
from unity_utils.unity_utils import Unity
import cv2
import time
import statistics
from Lane import *
import numpy as np
import argparse
from process import *

time_list = [0]*10

def stdev_list(list, point):
    list.append(point)
    del list[:-10]
    #print(list)
    avg = sum(list)/len(list)
    avg += statistics.stdev(list)
    return avg

def full_img(left_image,right_image):
    left_image = left_image[:,0:320,:]
    right_image = right_image[:,40:360,:]
    img_full=cv2.resize(left_image,dsize=(640,150))
    img_full[:,:320,:]=left_image
    img_full[:,320:640,:]=right_image[:,:,:]
    return img_full

def final_img():
    left_image, right_image = unity_api.get_images()
    for i in range(3):
        left_image1, right_image1 = unity_api.get_images()
        left_image = cv2.addWeighted(left_image,1,left_image1,1.0,0.0)
        right_image = cv2.addWeighted(right_image,1,right_image1,1.0,0.0)
        i += 1
    
    img_full = full_img(left_image,right_image)
    #cv2.imshow('DeNoise',img_full)
    unity_api.show_images(left_image, right_image)
    return img_full


def processing(image):
    left_image, right_image = unity_api.get_images()
    display_img = full_img(left_image,right_image)
    ############# LINES ####################
    binary_image =  binary_pipeline(image)
    bird_view, inverse_perspective_transform =  warp_image(binary_image)
    left_fit, right_fit = track_lanes_initialize(bird_view)
    left_fit, right_fit = check_fit_duplication(left_fit,right_fit)
    center_fit, left_fit, right_fit = find_center_line_and_update_fit(image,left_fit,right_fit) # update left, right line
    colored_lane, center_line = lane_fill_poly(bird_view,image,center_fit,left_fit,right_fit, inverse_perspective_transform)
    cv2.imshow("lane",colored_lane)
    # cv2.imshow("image_cp_ts",image_cp_ts)
    speed_current, steer_angle = get_speed_angle(center_line)
    #if traffic_sign is None and flag_ts and (steer_angle >= 20 or steer_angle <= -20):
    #   return status_ts[0], status_ts[1]

    edge_img = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)
    final_img = cv2.addWeighted(display_img,1.0,edge_img,1.0,0.0)
    cv2.imshow('final_img',final_img)
    return int(speed_current), int(steer_angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mix port of car')
    parser.add_argument('--port', type=int, default=11000, metavar='PORT',help='mix port of car 1')
    args=parser.parse_args()

    unity_api = Unity(args.port)
    unity_api.connect()

    while True:
        start_time = time.time()
        img_full = final_img()

        s, a  = processing(img_full)
        print('Speed: ', s, "\tAngle: ",a)

        finalTime = int(stdev_list(time_list,1/(time.time() - start_time)))
        print("FPS: ", finalTime)
        cv2.imshow('Full image',img_full)