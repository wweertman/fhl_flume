# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:34:08 2020

@author: wlwee
"""
import os, time, cv2
from datetime import datetime
from pypylon import pylon
from pypylon import genicam

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def get_root_time (grabResult):    
    if grabResult.GrabSucceeded():
        t1 = time.time()
        ts = grabResult.TimeStamp * 1.e-9
        root_time = t1 - ts
        print('started at ' + str(time.ctime(t1)))
        grabResult.Release()
    else:
        print('failed to grab from camera')
    return root_time

def parameterize_cameras(number_of_cameras, list_of_setting_lists):
    
    '''
    Int: number_of_cameras, number of cameras attached to computer
    
    List: list_of_setting_lists, [[pixel_format, exposure, gain, fps],[...], ... ,[...]] sets capture parameters for each camera
    must have the same number of sub-lists as 'number_of_cameras' 
    sublists must have the format of [pixel_format, exposure, gain, fps]
    
    '''
    
    #Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    #Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RUNTIME_EXCEPTION("No camera present.")

    #Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), number_of_cameras))
    #l = cameras.GetSize()
    try:
        for i, cam in enumerate(cameras):
            
            a = list_of_setting_lists[i]
            pixel_format = a[0]
            exposure = a[1]
            gain = a[2]
            fps = a[3]
            
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            
            cam.Open()
            
            print('camera ' + str(i) + ' model name')
            cam_info = cam.GetDeviceInfo().GetModelName(), "-", cam.GetDeviceInfo().GetSerialNumber()
            cam_info = cam_info[0] + cam_info[1] + cam_info[2]
            print(cam_info)
            
            cam.PixelFormat.SetValue(pixel_format)         
            cam.ExposureTime.SetValue(exposure)
            cam.Gain.SetValue(gain)
            cam.AcquisitionFrameRateEnable.SetValue(True);
            cam.AcquisitionFrameRate.SetValue(fps)
            
            print('pixel format, exposure, gain, fps')
            spec = pixel_format + ', ' + str(exposure) + ', ' + str(gain) + ', ' + str(fps)
            print(spec)
            print('resulting frame rate')
            cam_info = str(cam.ResultingFrameRate.GetValue())
            print(cam_info)
            
            # Print the model name of the camera.
            cam_info = cam.GetDeviceInfo().GetModelName()
            cam.Close()
    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", str(e))
        cameras.Close()
    
    return cameras

def grab_root_time_list (cameras):
    
    l = cameras.GetSize()
    cameras.StartGrabbing()
    root_time_lst = []
    
    for i in range(0,l):
        
        if not cameras.IsGrabbing():
            print('cameras failed to grab')
            break
        
        grabResult = cameras.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            t1 = time.time()
            ts = grabResult.TimeStamp * 1.e-9
            root_time = t1 - ts
            print('started at ' + str(time.ctime(t1)))
            grabResult.Release()
        else:
            print('failed to grab from camera')
    
        root_time_lst.append(root_time)
    
    cameras.StopGrabbing()    
    
    return root_time_lst

target_dir = r'C:\Users\wlwee\Documents\python\fhl_flume_single_target_experiment\CODE\camera_scripts\test_multi_camera_capture'
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)
    
cam1_name = 'cam1'
cam2_name = 'cam2'    

vid1_dir = target_dir + '/' + cam1_name
vid2_dir = target_dir + '/' + cam2_name

if os.path.exists(vid1_dir) != True:
    os.mkdir(vid1_dir)
if os.path.exists(vid2_dir) != True:
    os.mkdir(vid2_dir)    
    
#[pixel_format, exposure, gain, fps]
cam_format_lst = [['Mono8', 10000, 2, 10],['Mono8', 10000, 2, 10]]
cameras = parameterize_cameras(2, cam_format_lst)

root_time_lst = grab_root_time_list(cameras)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

video_length = 10 #minutes
vid_ext = 'mp4'

vid1_counter, vid2_counter = 0, 0
vid1_numerator, vid2_numerator = 0, 0

try:    
    cameras.StartGrabbing()
    
    print('starting to record from cameras')
    print('crtl + c to end')
    
    while True:
        if not cameras.IsGrabbing():
            break
            
        grabResult = cameras.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)        
        cameraContextValue = grabResult.GetCameraContext() #gets int for cameras number
        
        img = converter.Convert(grabResult)
        img = img.GetArray()
        grabTime = grabResult.TimeStamp * 1.e-9
        
        if cameraContextValue == 0:
            
            t = root_time_lst[cameraContextValue] + grabTime
            
            cv2.putText(img, #numpy array on which text is written
                        time.ctime(t), #text
                        (25,25), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        1, #font size
                        (209, 80, 0, 255), #font color
                        3)
            
            cv2.imshow(cam1_name, img)
            
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            if vid1_numerator == 0:
                vid1_subdir = vid1_dir + '/' + cam1_name + '_' + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_' + str(vid1_counter)
                if os.path.exists(vid1_subdir) != True:
                    os.mkdir(vid1_subdir)
                vid1_counter = vid1_counter + 1
                
                dim = (grabResult.GetWidth(), grabResult.GetHeight())
                video1_name = vid1_subdir + '/' + cam1_name + '_' + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_' + str(vid1_counter) + '.' + vid_ext 
                video1 = cv2.VideoWriter(video1_name,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        cam_format_lst[cameraContextValue][3],
                                        dim)
                
            if vid1_numerator >= video_length * cam_format_lst[cameraContextValue][3] * 60:
                video1.release()
                vid1_numerator = 0
            else:
                vid1_numerator = vid1_numerator + 1
            
        if cameraContextValue == 1:            
            t = root_time_lst[cameraContextValue] + grabTime
            
            cv2.putText(img, #numpy array on which text is written
                        time.ctime(t), #text
                        (25,25), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        1, #font size
                        (209, 80, 0, 255), #font color
                        3)
            
            cv2.imshow(cam2_name, img)
            
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            if vid2_numerator == 0:
                vid2_subdir = vid1_dir + '/' + cam2_name + '_' + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_' + str(vid2_counter)
                if os.path.exists(vid2_subdir) != True:
                    os.mkdir(vid2_subdir)
                vid2_counter = vid2_counter + 1
                
                dim = (grabResult.GetWidth(), grabResult.GetHeight())
                video2_name = vid2_subdir + '/' + cam2_name + '_' + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_' + str(vid2_counter) + '.' + vid_ext 
                video2 = cv2.VideoWriter(video2_name,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        cam_format_lst[cameraContextValue][3],
                                        dim)
                
            if vid2_numerator >= video_length * cam_format_lst[cameraContextValue][3] * 60:
                video2.release()
                vid1_numerator = 0
            else:
                vid2_numerator = vid2_numerator + 1   
            
        grabResult.Release()
        del img
        
except KeyboardInterrupt:
    print('keyboard interupt')
    video1.release()
    video2.release()
    cameras.Close()
    cv2.destroyAllWindows()
except SystemError:
    print('system error, likely due to instant camera array')
    video1.release()
    video2.release()
    cameras.Close()
    cv2.destroyAllWindows()
    










