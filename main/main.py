from turtle import speed
from unity_utils.unity_utils import Unity
import cv2
import argparse
from process import *
from PIL import Image
from signboard_detect import *
from pid import PID

PID_controller = PID(sampletime=0.05,kp=10,ki=0.3,kd=1,out_min=-25,out_max=25)

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
    #unity_api.show_images(left_image, right_image)
    return img_full


def processing(image):
    global state
    left_image, right_image = unity_api.get_images()
    display_img = full_img(left_image,right_image)
    binary_image =  binary_pipeline(image)
    edge_img = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)
    final_img = cv2.addWeighted(display_img,1.0,edge_img,1.0,0.0)
    ############# LINES ####################
    bird_view, inverse_perspective_transform =  warp_image(binary_image)
    left_fit, right_fit = track_lanes_initialize(bird_view)

    center_fit, left_fit_update, right_fit_update = find_center_line_and_update_fit(image,left_fit, right_fit) # update left, right line
    #print('\nleft fit',np.sum(left_fit_update),'\tright fit',np.sum(right_fit_update))
    #print('center',center_fit)
    colored_lane, center_line = lane_fill_poly(bird_view,edge_img,center_fit,left_fit_update,right_fit_update, inverse_perspective_transform)
    #cv2.imshow("lane",colored_lane)
    cv2.imshow("bird_view",bird_view)
    state = detect_color(display_img)
    #print(state)
    speed_current, steer_angle = get_speed_angle(center_line,state)
    #if traffic_sign is None and flag_ts and (steer_angle >= 20 or steer_angle <= -20):

    #edge_img = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2RGB)
    #final_img = cv2.addWeighted(display_img,1.0,edge_img,1.0,0.0)
    unity_api.show_images(left_image, right_image)
    cv2.imshow('final_img',final_img)
    return float(speed_current), float(steer_angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mix port of car')
    parser.add_argument('--port', type=int, default=11000, metavar='PORT',help='mix port of car 1')
    args=parser.parse_args()
    unity_api = Unity(args.port)
    unity_api.connect()
    data = unity_api.set_speed_angle(0, 0)

    state = 0
    while True:
        img_full = final_img()
        res = []

        speedcal, angcal  = processing(img_full)
        speedcal = stdev_list(speed_list,speedcal)
        #angcal = stdev_list(ang_list,angcal)
        #print('Speed: ', speedcal, "\tAngle: ",angcal,'\n')
        
        
        speedcal = speed_control(data['Speed'], speedcal)
        angcal = PID_controller.calc(data['Angle'],angcal)

        data = unity_api.set_speed_angle(speedcal, angcal)
        print('Speed: ', data['Speed'], "\tAngle: ",angcal,'\n')
        #print(data)
        finalTime = int(stdev_list(time_list,1/(time.time() - start_time)))
