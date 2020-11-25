# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:51:51 2020

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

def stream_video (cam, root_time):
    try:
        print('starting to stream from cam')
        print('crtl + c to end')
        while cam.IsGrabbing():
            grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)        
            grabTime = grabResult.TimeStamp * 1.e-9
            t = root_time + grabTime
            
            img = converter.Convert(grabResult)
            img = img.GetArray()
            
            cv2.putText(img, #numpy array on which text is written
                        time.ctime(t), #text
                        (25,25), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        1, #font size
                        (209, 80, 0, 255), #font color
                        3) #font stroke
            
            img_resize = ResizeWithAspectRatio(img, width=800)
            
            cv2.imshow(cam_info, img_resize)
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            grabResult.Release()
            del img
            del img_resize
        
    except KeyboardInterrupt:
        print('keyboard interupt')
        cv2.destroyAllWindows()
    except SystemError:
        print('system error, likely due to instant camera array')
        cv2.destroyAllWindows()

def record_video(cam, target_dir, vid_name, vid_ext, fps, resize = 0.75, save_resize = False):
    try:
        print('starting to record from cam')
        print('crtl + c to end')
        grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            pass
        else:
            print('grabResult failed')
        
        vname = vid_name + datetime.today().strftime('_%Y_%m_%d__%H_%M_%S_') + 'f' + str(fps) 
        vid_count, vid_numerator = 0, 0
        save_dir = target_dir + '/' + vid_name + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_vid_' + str(vid_count)
        os.mkdir(save_dir)
        video_name = save_dir + '/' + vname + '_c' + str(vid_count) + '.' + vid_ext
        if save_resize == True:
            dim = (int(resize*grabResult.GetWidth()), int(resize*grabResult.GetHeight()))
        else: 
            dim = (grabResult.GetWidth(), grabResult.GetHeight())
        resize_width = int(resize * grabResult.GetWidth())
        resize_height = int(resize * grabResult.GetHeight())
        print('resized view window to: ' +  '(' + str(resize_width) + ', ' + str(resize_height) + ')')
        print('img dim: ' + str(dim))
        video = cv2.VideoWriter(video_name,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                dim)
        while cam.IsGrabbing():                       
            grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)        
            grabTime = grabResult.TimeStamp * 1.e-9
            t = root_time + grabTime
            
            img = converter.Convert(grabResult)
            img = img.GetArray()
            
            cv2.putText(img, #numpy array on which text is written
                        time.ctime(t), #text
                        (25,25), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        1, #font size
                        (209, 80, 0, 255), #font color
                        3) #font stroke
            
            img_resize = ResizeWithAspectRatio(img, width=resize_width)
            
            cv2.imshow(cam_info, img_resize)
            
            if save_resize == True:
                video.write(img_resize)
            else:
                video.write(img)
                
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            if vid_numerator >= video_length * fps * 60:
                video.release()
                vid_count = vid_count + 1
                vid_numerator = 0
                save_dir = target_dir + '/' + vid_name + datetime.today().strftime('_%m_%d__%H_%M_%S') + '_vid_'  + str(vid_count)
                os.mkdir(save_dir)
                vname = vid_name + datetime.today().strftime('_%Y_%m_%d__%H_%M_%S_') + 'f' + str(fps) 
                video_name = save_dir + '/' + vname + '_c' + str(vid_count) + '.' + vid_ext
                video = cv2.VideoWriter(video_name,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps,
                                        dim)
            else:
                vid_numerator = vid_numerator + 1
                
            grabResult.Release()
            del img
            del img_resize
        
    except KeyboardInterrupt:
        print('keyboard interupt')
        video.release()
        cam.Close()
        cv2.destroyAllWindows()
    except SystemError:
        print('system error, likely due to instant camera array')
        video.release()
        cv2.destroyAllWindows()
        cam.Close()

def initiate_and_setup_cam(pixel_format,
                           exposure,
                           gain,
                           fps):
    
    os.environ["PYLON_CAMEMU"] = "3"
    
    try:

        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        cam.Open()
        
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
        
        cam_info = cam.GetDeviceInfo().GetModelName()
        print("Using device " + cam_info)
        
        cam.StartGrabbing()
        
    except genicam.GenericException as e:
        # Error handling
        print("An exception occurred.", str(e))
        cam.Close()
    
    return cam
    
def get_root_time (cam):    
    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        t1 = time.time()
        ts = grabResult.TimeStamp * 1.e-9
        root_time = t1 - ts
        print('started at ' + str(time.ctime(t1)))
        grabResult.Release()
    else:
        print('failed to grab from camera')
    return root_time

target_dir = r'C:\Users\wlwee\Documents\python\record_w_basler_camera\DATA\muusoctopus_observation'
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)

vid_name = 'test'
vid_ext = 'mp4'
video_length = 30 #minutes

fps = 5
exposure = 5000
gain = 1
pixel_format = 'Mono8'

cam = initiate_and_setup_cam(pixel_format = pixel_format,
                               exposure = exposure, 
                               gain = gain,
                               fps = fps)
cam_info = cam.GetDeviceInfo().GetModelName()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

root_time = get_root_time(cam)

stream_video(cam, root_time)

record_video(cam, target_dir, vid_name, vid_ext, fps, resize = 0.5, save_resize = True)
