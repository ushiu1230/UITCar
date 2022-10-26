from turtle import speed
from unity_utils.unity_utils import Unity
import cv2
import time
import statistics
from Lane import *
import numpy as np
import argparse
from process import *
from PIL import Image

speed_list = [0]*10
ang_list = [0]*10
time_list = [0]*10

def stdev_list(list, point):
    list.append(point)
    del list[:-8]
    #print(list)
    avg = sum(list)/len(list)
    avg += statistics.stdev(list)
    return avg

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def full_img(left_image,right_image):
    left_image = left_image[:,0:300,:]
    right_image = right_image[:,0:300,:]
    img_full=cv2.resize(left_image,dsize=(610,150))
    img_full[:,:300,:]=left_image
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img line', left_gray[:,300,:])
    left_list = list(left_gray[:,299])
    right_list = list(right_gray[:,1])
    if sum(left_list) <= sum(right_list):
        for i in range(10):
            img_full[:,300+i,:] = right_image[:,1,:]
    else:
        for i in range(10):
            img_full[:,300+i,:] = left_image[:,299,:]
    img_full[:,310:610,:]=right_image[:,:,:]
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
    binary_image =  binary_pipeline(image)
    edge_img = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)
    final_img = cv2.addWeighted(display_img,1.0,edge_img,1.0,0.0)
    ############# LINES ####################
    bird_view, inverse_perspective_transform =  warp_image(binary_image)
    left_fit, right_fit = track_lanes_initialize(bird_view)
    #print(np.sum(left_fit),np.sum(right_fit),'\n')
    #left_fit1, right_fit1 = fix_laneline(left_fit,right_fit)
    center_fit, left_fit_update, right_fit_update = find_center_line_and_update_fit(image,left_fit, right_fit) # update left, right line
    print('\nleft fit',np.sum(left_fit_update),'\tright fit',np.sum(right_fit_update))
    #print('center',np.sum(center_fit))
    colored_lane, center_line = lane_fill_poly(bird_view,edge_img,center_fit,left_fit_update,right_fit_update, inverse_perspective_transform)
    cv2.imshow("lane",colored_lane)
    cv2.imshow("bird_view",bird_view)
    speed_current, steer_angle = get_speed_angle(center_line,left_fit, right_fit_update)
    #if traffic_sign is None and flag_ts and (steer_angle >= 20 or steer_angle <= -20):
    #   return status_ts[0], status_ts[1]

    #edge_img = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)
    #final_img = cv2.addWeighted(display_img,1.0,edge_img,1.0,0.0)
    #cv2.imshow('final_img',final_img)
    return int(speed_current), float(steer_angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mix port of car')
    parser.add_argument('--port', type=int, default=11000, metavar='PORT',help='mix port of car 1')
    args=parser.parse_args()

    unity_api = Unity(args.port)
    unity_api.connect()

    kp = 0.7
    ki = 0
    kd = 0.1
    pid = PID(kp,ki,kd, setpoint= 0)

    pid.output_limits = (-15,15)
    while True:
        start_time = time.time()
        img_full = final_img()

        speedcal, angcal  = processing(img_full)
        angcal = pid(-(angcal))
        speedcal = stdev_list(speed_list,speedcal)
        angcal = stdev_list(ang_list,angcal)
        print('Speed: ', speedcal, "\tAngle: ",angcal)

        unity_api.set_speed_angle(speedcal, angcal)
        #print(data)
        finalTime = int(stdev_list(time_list,1/(time.time() - start_time)))
        #print("FPS: ", finalTime)
        #cv2.imshow('Full image',img_full)