# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:32:37 2020

@author: wlwee
"""

#Grab_MultipleCameras.cpp
#============================================================================
#This sample illustrates how to grab and process images from multiple cameras
#using the CInstantCameraArray class. The CInstantCameraArray class represents
#an array of instant camera objects. It provides almost the same interface
#as the instant camera for grabbing.
#The main purpose of the CInstantCameraArray is to simplify waiting for images and
#camera events of multiple cameras in one thread. This is done by providing a single
#RetrieveResult method for all cameras in the array.
#Alternatively, the grabbing can be started using the internal grab loop threads
#of all cameras in the CInstantCameraArray. The grabbed images can then be processed by one or more
#image event handlers. Please note that this is not shown in this example.
#============================================================================

import os

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys
import time

#enter the trial name here, date time will automatically be appended to the folder name
test_name = 'dual_camera_compression_test'
#enter the folder where you want the trial to be stored, make sure you end with a '/'
#'D:/FHL_Flume/Dual_Camera_Data/'
#'C:/Users/wlwee/Documents/python/fhl_pypylon/DATA/dual_camera_test/'
folder_path = 'C:/Users/wlwee/Documents/python/fhl_pypylon/DATA/dual_camera_test/'

#cheap way to drop fps, better off adjusting c_fps, time.sleep() does not stabilize the script
camera_sleep_time = 0

#adjusting exposure and gain should be done in the pylon viewer then set here
#Mono8 or Mono12 are the two pixel formats to consider, Mono8 takes less data
c_exposure, c_gain, c_fps = 20000.0, 5.0, 100.0
c_pixel_format = "Mono8"

#replace with False if you do not wish to compress images immediatly after the trial
compress_raw_images = True
#replace with False if you do not wish to replace the raw images after the trial, only if compress_raw_images = 'yes'
remove_raw_images = True
#if you want to compress to an external hard drive after capturing to your SSD set to True
compress_to_external_harddrive = True
external_hard_drive_path = 'D:/FHL_Flume/Dual_Camera_Data/'

#jpeg quality modification, 100 - i * 10
#0 is the highest quality
i = 0
ipo = pylon.ImagePersistenceOptions()
q = 100 - i * 10
ipo.SetQuality(q)

#raw image codec of .tiff works right now
image_codec = 'tiff'
image_codec_compress = ''

if image_codec == 'tiff':
    #raw image codec of .jpeg works right now and nothing else
    image_codec_compress = 'jpeg'

name_time = ''
for time_part in time.ctime().split(' '): 
    name_time = name_time + '_' + time_part
    
name_time = name_time.split(':')
names_time = ''
for time_part in name_time: 
    names_time = names_time + '_' + time_part
    
test_name = test_name + names_time

trial_dir = folder_path + test_name + '/'
codec_path = trial_dir + image_codec + '/'

if compress_to_external_harddrive == False:
    compression_path = trial_dir + image_codec_compress + '/'
else:
    folder_external_hard_drive = external_hard_drive_path + test_name + '/'
    if os.path.isdir(folder_external_hard_drive) == False:
        print()
        print('exporting compression to external hard drive after capture')
        print('expect delays after capture!')
        os.mkdir(folder_external_hard_drive)
        print('making dir: ' + folder_external_hard_drive)
    compression_path = folder_external_hard_drive + image_codec_compress + '/'

if os.path.isdir(trial_dir) == False:
    
    print()
    os.mkdir(trial_dir)
    print('making dir: ' + trial_dir)
    os.mkdir(codec_path)
    print('making dir: ' + codec_path)
    os.mkdir(compression_path)
    print('making dir: ' + compression_path)

    path_camera_0 = codec_path + 'camera_0/'
    path_camera_1 = codec_path + 'camera_1/'
    
    os.mkdir(path_camera_0)
    print('making dir: ' + path_camera_0)
    os.mkdir(path_camera_1)
    print('making dir: ' + path_camera_1)

elif os.path.isdir(trial_dir) == True: 
    print('trial_dir exists')
else:
    print('incorrect filepath')
    
print()

#Limits the amount of cameras used for grabbing.
#It is important to manage the available bandwidth when grabbing with multiple cameras.
#This applies, for instance, if two GigE cameras are connected to the same network adapter via a switch.
#To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay
#parameter can be set for each GigE camera device.
#The "Controlling Packet Transmission Timing with the Interpacket and Frame Transmission Delays on Basler GigE Vision Cameras"
#Application Notes (AW000649xx000)
#provide more information about this topic.
#The bandwidth used by a FireWire camera device can be limited by adjusting the packet size.
maxCamerasToUse = 2

#The exit code of the sample application.
exitCode = 0

time_camera_0 = []
time_camera_1 = []
fps_img_save = []

t = 0

#img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
def pylon_image_save_jpeg (f, ip, ig):
    f = f + ".jpeg" 
    ig.Save(pylon.ImageFileFormat_Jpeg, f, ip)
    
def pylon_image_save_tiff (f, ig):
    f = f + ".tiff"
    ig.Save(pylon.ImageFileFormat_Tiff,f)

try:

    #Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    #Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RUNTIME_EXCEPTION("No camera present.")

    #Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    l = cameras.GetSize()

    #Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        
        cam.Open()
        
        print('camera ' + str(i) + ' model name')
        print(cam.GetDeviceInfo().GetModelName(), "-", cam.GetDeviceInfo().GetSerialNumber())
        
        cam.PixelFormat.SetValue(c_pixel_format)         
        cam.ExposureTime.SetValue(c_exposure)
        cam.Gain.SetValue(c_gain)
        cam.AcquisitionFrameRateEnable.SetValue(True);
        cam.AcquisitionFrameRate.SetValue(c_fps)
        print('pixel format, exposure, gain, fps')
        print(c_pixel_format + ', ' + str(c_exposure) + ', ' + str(c_gain) + ', ' + str(c_fps))
        print('resulting frame rate')
        print(cam.ResultingFrameRate.GetValue())
        
        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())
        cam.Close()
        print()

    #Starts grabbing for all cameras starting with index 0. The grabbing
    #is started for one camera after the other. That's why the images of all
    #cameras are not taken at the same time.
    #However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    #According to their default configuration, the cameras are
    #set up for free-running continuous acquisition.
    
    cameras.StartGrabbing()
    
    imageWindow = pylon.PylonImageWindow()
    imageWindow.Create(1)
    
    img = pylon.PylonImage()
    
    t_w = time.perf_counter()
    
    try:
        
        print('watching live feed, crtl + c to record')
        print()
        
        while True:
            
            if not cameras.IsGrabbing():
                break
            
            grabResult = cameras.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            cameraContextValue = grabResult.GetCameraContext()
            
            if grabResult.GrabSucceeded():
                if cameraContextValue == 0:
                    imageWindow.SetImage(grabResult)
                    imageWindow.Show()
            else:
                print("Error: ",
                      grabResult.ErrorCode)  #grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError   
            
            #img.Release()
            grabResult.Release()
            
    except KeyboardInterrupt:
        print('keyboard interupt')
        print('moving to record')
        print()
        
    except SystemError:
        print('system error, likely due to instant camera array')
        print('moving to record')
        print()
    
    t_w = time.perf_counter() - t_w
    print('watched for: ' + str(t_w))
    
    t_s = time.perf_counter()
    
    try:
        
        print()
        print('starting to record')
        print('crtl + c to end recording')
        print()
        
        while True:
            
            if not cameras.IsGrabbing():
                break
            
            grabResult = cameras.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            cameraContextValue = grabResult.GetCameraContext()
            
            if grabResult.GrabSucceeded():
                imageWindow.SetImage(grabResult)
                if cameraContextValue == 0:
                    imageWindow.Show()
            else:
                print("Error: ",
                      grabResult.ErrorCode)  # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError   
            
            
            
            img.AttachGrabResultBuffer(grabResult)
            
            the_time = str(time.perf_counter()).split('.')
            the_time = the_time[0] + '_' + the_time[1]
            
            if cameraContextValue == 0:
                filename = path_camera_0 + the_time
                time_camera_0.append(time.perf_counter())
            else:
                filename = path_camera_1 + the_time 
                time_camera_1.append(time.perf_counter())
            
            t = time.perf_counter()
            
            if image_codec == 'jpeg':
                pylon_image_save_jpeg(filename, ipo, img)
            if image_codec == 'tiff':
                pylon_image_save_tiff(filename, img)
            
            if not imageWindow.IsVisible():
                cameras.StopGrabbing()
             
             
            img.Release()
            grabResult.Release()
            time.sleep(camera_sleep_time)
            
            fps_img_save.append(1/((time.perf_counter() - t)))
    
    except KeyboardInterrupt:
        print('keyboard interupt')
        cameras.Close()
        imageWindow.Close()
        exitCode = 2
    
    except SystemError:
        print('system error, likely due to instant camera array')
        cameras.Close()
        imageWindow.Close()
        exitCode = 3
    
except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", str(e))
    cameras.Close()
    imageWindow.Close()
    exitCode = 1

t_s = time.perf_counter() - t_s
print('time length trial: ' + str(t_s))
print()

#end recording script
camera_0_fps = []
camera_1_fps = []

try:
    del time_camera_0[-1]
except IndexError():
    print('time_camera_0 does not exist')

#this bit will spit out the fps for each camera to check the code
#print('camera_0 delta time (seconds), 1/(time_camera_0[d + 1] - time_camera_0[d]')
for d in range(0, len(time_camera_0),2):
    try:
        #print(time_camera_0[d])
        #print(time_camera_0[d + 1])
        fps = 1/(time_camera_0[d + 1] - time_camera_0[d])
        #print(str(fps))
        camera_0_fps.append(fps)
    except IndexError:
        print('IndexError end of odd camera_0 list') 

#print('camera_1 delta time (seconds), 1/(time_camera_0[d + 1] - time_camera_0[d]')
for d in range(0, len(time_camera_1),2):
    try:
        #print(time_camera_1[d])
        #print(time_camera_1[d + 1])
        fps = 1/(time_camera_1[d + 1] - time_camera_1[d])
        #print(str(fps))
        camera_1_fps.append(fps)
    except IndexError:
        print('IndexError end of odd camera_1 list') 

from matplotlib import pyplot as plt
import statistics 

plt.subplot(1, 3, 1)
plt.plot(range(0,len(camera_0_fps)),
         camera_0_fps,
         label = 'camera_0')
plt.plot(range(0,len(camera_1_fps)),
         camera_1_fps,
         label = 'camera_1')
plt.ylabel('FPS')
plt.title('capture rate')
plt.legend(loc='lower right', borderaxespad=0.1)
plt.subplot(1, 3, 2)
plt.plot(range(0,len(fps_img_save)),
         fps_img_save,
         label = 'save_fps')
plt.title('save rate')
plt.legend(loc='lower right', borderaxespad=0.1)

print()
print('mean fps camera_0: ' + str(statistics.mean(camera_0_fps)))
print('mean fps camera_1: ' + str(statistics.mean(camera_1_fps)))
print('mean fps save fps: ' + str(statistics.mean(fps_img_save)))
print()

from PIL import Image
import glob

fps_img0_compress = []
fps_img1_compress = []

def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "_K", "_M", "_G", "_T", "_P", "_E", "_Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

if compress_raw_images == True:

    try:
        
        if os.path.isdir(compression_path) == True:
         
            compress_path_camera_0 = compression_path + 'camera_0/'
            os.mkdir(compress_path_camera_0)
            compress_path_camera_1 = compression_path + 'camera_1/'
            os.mkdir(compress_path_camera_1)
            
            print ('making dir: ' + compression_path)
            print ('making dir: ' + compress_path_camera_0)
            print ('making dir: ' + compress_path_camera_1)
        else:
            print('incorrect filepath')
        
        print()
        print('compressing camera_0 images')
        
        camera_0_imgs = glob.glob(path_camera_0 + '*.' + image_codec)
        camera_1_imgs = glob.glob(path_camera_1 + '*.' + image_codec)
        
        t_c = time.perf_counter()
        
        for raw_img in camera_0_imgs:
            
            t = time.perf_counter()
            path_compressed_img = compress_path_camera_0 + os.path.basename(raw_img).split('.')[0] + '.' + image_codec_compress
            
            im = Image.open(raw_img)
            im.thumbnail(im.size)
            
            #quality = q if for compressing to .jpeg
            im.save(path_compressed_img,
                    image_codec_compress.upper(),
                    quality = q)
            
            im.close()
            fps_img0_compress.append(1/(time.perf_counter() - t))
        
        print()
        print('compressing camera_1 images')
        
        for raw_img in camera_1_imgs:
            
            t = time.perf_counter()
            path_compressed_img = compress_path_camera_1 + os.path.basename(raw_img).split('.')[0] + '.' + image_codec_compress
            
            im = Image.open(raw_img)
            im.thumbnail(im.size)
            
            #quality = q if for compressing to .jpeg
            im.save(path_compressed_img,
                    image_codec_compress.upper(),
                    quality = q)
            
            im.close()
            fps_img1_compress.append(1/(time.perf_counter() - t))
        
        print()
        plt.subplot(1, 3, 3)
        plt.plot(range(0,len(fps_img0_compress)),
                 fps_img0_compress,
                 label = 'comp_0')
        plt.plot(range(0,len(fps_img1_compress)),
                 fps_img1_compress,
                 label = 'comp_1')
        plt.legend(loc='lower right', borderaxespad=0.1)
        plt.title('compress rate')
        
        print('mean compress rate camera_0: ' + str(statistics.mean(fps_img0_compress)))
        print('mean compress rate camera_1: ' + str(statistics.mean(fps_img1_compress))) 
        
        print()
        t_c = time.perf_counter() - t_c
        print('time taken to compress: ' + str(t_c))
        print('compression time/trial time: ' + str(t_c/t_s))
        
        if remove_raw_images == True:
            
            t = time.perf_counter()
            
            import shutil
            
            print()
            print('removing raw images')
            print()
            
            raw_dir_size = get_size_format(get_directory_size(codec_path))
            print('size of raw images: ' + raw_dir_size)        
            raw_size_rate = float(raw_dir_size.split('_')[0])/t_s
            print('rate of creation raw images: ' + str(raw_size_rate) + ' ' + raw_dir_size.split('_')[1] + '/s')
            
            shutil.rmtree(codec_path)
            print()
            print('time taken to count and delete raw images: ' + str(time.perf_counter() - t))
            print()
            
        comp_dir_size = get_size_format(get_directory_size(compression_path))
        print('size of compressed images: ' + comp_dir_size)
        comp_size_rate = float(comp_dir_size.split('_')[0])/t_s
        print('rate of creation comp images: ' + str(comp_size_rate) + ' ' + comp_dir_size.split('_')[1] + '/s')
        
    except KeyboardInterrupt:
        print('keyboard interupt during compression')
        
print()
plt.show()

sys.exit(exitCode)