# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:28:21 2020

@author: wlwee
"""
import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import time
import shutil
import random


test_name = 'script_test_9_4_observation'
folder_path = 'W:/FHL_FLUME/Slaanesh/'
external_hard_drive_path = 'D:/FHL_Flume/slaanesh/'

mylist = [0, 0, 0]
minVal, maxVal = 0, 1            
index = random.randrange(len(mylist))
mylist[index] = 1

name_time = ''
for time_part in time.ctime().split(' '): 
    name_time = name_time + '_' + time_part
    
name_time = name_time.split(':')
names_time = ''
for time_part in name_time: 
    names_time = names_time + '_' + time_part
    
test_name = test_name + names_time

c_exposure, c_gain, c_fps = 20000.0, 5.0, 7.0
c_pixel_format = "Mono8"

#jpeg quality modification, 100 - i * 10
#0 is the highest quality
i = 0
ipo = pylon.ImagePersistenceOptions()
q = 100 - i * 10
ipo.SetQuality(q)

#raw image codec of .tiff works right now
image_codec = 'jpeg'
image_codec_compress = 'jpeg'

trial_dir = folder_path + test_name + '/'
codec_path = trial_dir + image_codec + '/'

folder_external_hard_drive = external_hard_drive_path + test_name + '/'
        
compression_path = folder_external_hard_drive + image_codec_compress + '/'

if os.path.isdir(trial_dir) == False:
    
    print()
    os.mkdir(trial_dir)
    print('making dir: ' + trial_dir)
    os.mkdir(codec_path)
    
    text_file_path = codec_path + test_name + '.txt' 
    text_file = open(text_file_path, "w")
    
    print()
    print('exporting compression to external hard drive after capture')
    print('expect delays after capture!')
    os.mkdir(folder_external_hard_drive)
    print('making dir: ' + folder_external_hard_drive)
    
    text_file.write('exporting compression to external hard drive after capture' + '\n')
    text_file.write('expect delays after capture!' + '\n')
    text_file.write('making dir: ' + folder_external_hard_drive + '\n')
    
    text_file.write('making dir: ' + trial_dir + '\n')
    print('making dir: ' + codec_path)
    text_file.write('making dir: ' + codec_path + '\n')
    os.mkdir(compression_path)
    print('making dir: ' + compression_path)
    text_file.write('making dir: ' + compression_path + '\n')
    
    path_camera_0_timelapse_external = compression_path + 'camera_0_timelapse/'
    path_camera_0_timelapse = codec_path + 'camera_0_timelapse/'
    
    os.mkdir(path_camera_0_timelapse)
    print('making dir: ' + path_camera_0_timelapse )
    text_file.write('making dir: ' + path_camera_0_timelapse + '\n')
    os.mkdir(path_camera_0_timelapse_external)
    print('making dir: ' + path_camera_0_timelapse_external)
    text_file.write('making dir: ' + path_camera_0_timelapse_external + '\n')
    
print()
text_file.write('\n')

exitCode = 0

def pylon_image_save_jpeg (f, ip, ig):
    f = f + ".jpeg" 
    ig.Save(pylon.ImageFileFormat_Jpeg, f, ip)
    
try:
    # Create an instant camera object with the camera device found first.
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    
    print('camera ' + str(i) + ' model name')
    text_file.write('camera ' + str(i) + ' model name' + '\n')
    cam_info = cam.GetDeviceInfo().GetModelName(), "-", cam.GetDeviceInfo().GetSerialNumber()
    cam_info = cam_info[0] + cam_info[1] + cam_info[2]
    print(cam_info)
    text_file.write(cam_info + '\n')
    
    cam.PixelFormat.SetValue(c_pixel_format)         
    cam.ExposureTime.SetValue(c_exposure)
    cam.Gain.SetValue(c_gain)
    cam.AcquisitionFrameRateEnable.SetValue(True);
    cam.AcquisitionFrameRate.SetValue(c_fps)
    print('pixel format, exposure, gain, fps')
    text_file.write('pixel format, exposure, gain, fps' + '\n')
    spec = c_pixel_format + ', ' + str(c_exposure) + ', ' + str(c_gain) + ', ' + str(c_fps)
    print(spec)
    text_file.write(spec + '\n')
    print('resulting frame rate')
    text_file.write('resulting frame rate' + '\n')
    cam_info = str(cam.ResultingFrameRate.GetValue())
    print(cam_info)
    text_file.write(cam_info + '\n')
    
    # Print the model name of the camera.
    cam_info = cam.GetDeviceInfo().GetModelName()
    print("Using device " + cam_info)
    text_file.write("Using device " + cam_info + '\n')
    
    cam.StartGrabbing()
    
    imageWindow = pylon.PylonImageWindow()
    imageWindow.Create(1)
    
    img = pylon.PylonImage()
    
    try:
        
        print()
        print('object array, E -> W, 1 = food, 0 = no food')
        print(str(mylist))
        
        text_file.write('\n')
        text_file.write('object array, E -> W, 1 = food, 0 = no food' + '\n')
        text_file.write(str(mylist) + '\n')
        
        print()
        print('watching live feed, crtl + c to record')
        print()
        text_file.write('\n')
        text_file.write('watching live feed, crtl + c to record' + '\n')
        text_file.write('\n')
        
        while True:
            
            if not cam.IsGrabbing():
                break
            
            grabResult = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            #grabTime = grabResult.TimeStamp
            cameraContextValue = grabResult.GetCameraContext()
            
            if grabResult.GrabSucceeded():
                imageWindow.SetImage(grabResult)
                imageWindow.Show()
            else:
                print("Error: ",
                      grabResult.ErrorCode)  #grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError   
            
            #img.Release()
            grabResult.Release()
        
    except KeyboardInterrupt:
        print('keyboard interupt')
        print('moving to record with both cameras')
        print()
        text_file.write('keyboard interupt' + '\n')
        text_file.write('moving to record with both cameras' + '\n')
        text_file.write('\n')
        
    except SystemError:
        print('system error, likely due to instant camera array')
        print('moving to record')
        print()
        text_file.write('system error, likely due to instant camera array' + '\n')
        text_file.write('moving to record with both cameras' + '\n')
        text_file.write('\n')
    
    try:
        
        time_recording_start = time.time()
        
        print('starting to record at fps = ' + str(c_fps))
        print('recording started at: ' + time.ctime(time_recording_start))
        print('crtl + c to end recording')
        print()
        text_file.write('\n')
        text_file.write('starting to record at fps = ' + str(c_fps) + '\n')
        text_file.write('crtl + c to end recording' + '\n')
        text_file.write('\n')
        
        while True:
            
            if not cam.IsGrabbing():
                break
            
            grabResult = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            grabTime = grabResult.TimeStamp * 1.e-9
            
            img.AttachGrabResultBuffer(grabResult)
            
            the_time = str(grabTime).split('.')
            the_time = the_time[0] + '_' + the_time[1]
            
            if grabResult.GrabSucceeded():
                imageWindow.SetImage(grabResult)
                imageWindow.Show()
                
                filename = path_camera_0_timelapse + the_time
                filename_external = path_camera_0_timelapse_external + the_time                    

                pylon_image_save_jpeg(filename, ipo, img)
                if c_fps <= 10.0:
                    pylon_image_save_jpeg(filename_external, ipo, img)
                    
            else:
                print("Error: ",
                      grabResult.ErrorCode)  # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError   

            if not imageWindow.IsVisible():
                cam.StopGrabbing()
             
            img.Release()
            grabResult.Release()
    
    except KeyboardInterrupt:
        print('keyboard interupt')
        print('moving to record one camera timelapse')
        text_file.write('keyboard interupt' + '\n')
        text_file.write('moving to record one camera timelapse' + '\n')
        text_file.write('\n')
        cam.Close()
        imageWindow.Close()
        exitCode = 2
    
    except SystemError:
        print('system error, likely due to instant camera array')
        print('moving to record with one camera')
        text_file.write('system error, likely due to instant camera array' + '\n')
        text_file.write('moving to record with one camera' + '\n')
        text_file.write('\n')
        cam.Close()
        imageWindow.Close()
        exitCode = 3
    
    time_recording_end = time.time()
    
except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", str(e))
    cam.Close()
    exitCode = 1

print()
print('recording started: ' + time.ctime(time_recording_start))
print('recording ended: ' + time.ctime(time_recording_end))
print('time length recording (min): ' + str((time_recording_end - time_recording_start)/60.0))
print('unix time started: ' + str(time_recording_start))
print('unix time ended: ' + str(time_recording_end))

text_file.write('recording started: ' + time.ctime(time_recording_start) + '\n')
text_file.write('recording ended: ' + time.ctime(time_recording_end) + '\n')
text_file.write('time length recording (min): ' + str((time_recording_end - time_recording_start)/60.0) + '\n')
text_file.write('unix time started: ' + str(time_recording_start) + '\n')
text_file.write('unix time ended: ' + str(time_recording_end) + '\n')

text_file.close()
shutil.copy2(text_file_path, 
             folder_external_hard_drive + os.path.basename(text_file_path))
sys.exit(exitCode)