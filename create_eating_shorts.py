# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:46:21 2020

@author: wlwee
"""
import os, glob
import pandas as pd
import statistics
import cv2
from shutil import copyfile

def get_fps(lst): 
    
    fps_nums = []
    for im in lst:
        fps_nums.append(im)
    fps_calc = []
    try:
        for n in range(0,len(fps_nums),2):
            fps_calc.append(1/(fps_nums[n + 1] - fps_nums[n]))
    except IndexError:
        print('fps_calc weird length')
       
    fps = round(statistics.mean(fps_calc),1)
    
    print('fps: ' + str(fps))
    print('num images: ' + str(len(lst)))
    print('length video: ' + str(len(lst)/fps/60) + ' min')

    return fps

def make_video(images, 
               video_name,
               fps):
        
    try: 
        frame = cv2.imread(images[0])
        frame_parameters = frame.shape
        
        height = frame_parameters[0]
        width = frame_parameters[1]
        
        video = cv2.VideoWriter(video_name, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (width, height))
        
        print()
        print('making video: ' + video_name)
        
        for image in images:
            img = cv2.imread(image)
            video.write(img)     
        
        cv2.destroyAllWindows()
        video.release()
    
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        video.release()

def closest(lst, K):  
    
    findex = min(range(len(lst)), key = lambda i: abs(lst[i] - K))
    fvalue = lst[findex]
    
    return findex, fvalue

def iterate_touch_csv (touch_dat):
    
    for i in touch_dat.itertuples():
        
        if i[2] == 1 and i[-1] == 1:
            
            return i[1]

def recombine_frame_str_to_float (frame_str):
    
    f = frame_str.split('_')[0] + '.' + frame_str.split('_')[1]
    f = float(f)
    
    return(f)

def sort_img_path_list (im_path):    
    im_float = recombine_frame_str_to_float(os.path.basename(im_path).split('.')[0])
    return im_float

def get_dir_frame_list (fdir):
    
    fs = glob.glob(fdir + '\\*.jpeg')
    fs = sorted(fs, key = sort_img_path_list)
    
    fs_head = []
    for fr in fs:
        f_str = os.path.basename(fr).split('.')[0]
        f_float = recombine_frame_str_to_float(f_str)
        fs_head.append(f_float)
    
    return fs, fs_head

pd.set_option('precision', 10)

path_img_dirs_main = r'E:\FHL_FLUME\khorne\three_targets_in_tank'
path_img_dirs = glob.glob(path_img_dirs_main + '\*\*')

path_img_dirs_eat = [edir for edir in path_img_dirs if edir.split('_')[-2] == 'eat']

path_extracted_video_dirs = r'W:\FHL_FLUME\three_target_experiment\eat_videos_food_to_mouth_five_min_back\khorne'

for adir in path_img_dirs_eat:
    
    print(adir)
    # this chunk makes/checks existance of the dirs where the short extracted videos will be going
    path_dir_step_1 = path_extracted_video_dirs + '\\' + adir.split('\\')[-2]
    
    if os.path.exists(path_dir_step_1) != True:
        os.mkdir(path_dir_step_1)
    
    path_dir_step_2 = path_dir_step_1 + '\\' + adir.split('\\')[-1]
    vid_name = path_dir_step_2 + '\\' + adir.split('\\')[-1] + '.mp4'
    csv_vid_name = path_dir_step_2 + '\\' + adir.split('\\')[-1] + '.csv'
    
    if os.path.exists(path_dir_step_2) != True:
        os.mkdir(path_dir_step_2)
    
    # this chunk checks if the touch csv was created
    stor_dir = adir + '\\storage_dir\\'
    three_tar_touch_csv_path = stor_dir + 'three_target_touch.csv'
    three_tar_touch_csv_copy_path = path_dir_step_2 + '\\three_target_touch.csv'
    
    if os.path.exists(three_tar_touch_csv_path) == True:
        dat_touch = pd.read_csv(three_tar_touch_csv_path)
        if os.path.exists(three_tar_touch_csv_copy_path) != True:
            copyfile(three_tar_touch_csv_path, 
                     three_tar_touch_csv_copy_path)
        else:
            print(three_tar_touch_csv_copy_path)
            print('exists')
    else:
        print(three_tar_touch_csv_path)
        print('touch csv does not exist')
        continue
    
    three_tar_location = stor_dir + 'three_target_location.csv'
    three_tar_location_csv_copy_path = path_dir_step_2 + '\\three_target_location.csv'
    
    if os.path.exists(three_tar_location) == True:
        dat_touch = pd.read_csv(three_tar_location)
        if os.path.exists(three_tar_location_csv_copy_path) != True:
            copyfile(three_tar_location, 
                     three_tar_location_csv_copy_path)
        else:
            print(three_tar_location_csv_copy_path)
            print('exists')
    else:
        print(three_tar_location)
        print('target location csv does not exist')
        continue
    
    # this chunk gets the frame where the food was brought to a mouth
    mouth_to_food = iterate_touch_csv(dat_touch)
    if isinstance(mouth_to_food, type(None)) != True:
        mouth_to_food = recombine_frame_str_to_float(mouth_to_food)
    else:
        continue
    
    # gets the path to the images and sorts them
    img_path_list, img_float_list = get_dir_frame_list(adir)
    
    # gets path and index of the frame where food is brought to the mouth
    index_food_touched = closest(img_float_list, mouth_to_food)
    
    # determines how far back we look from when the food is brought to the mouth
    look_back_time = 60 * 5 #seconds
    
    # gets path and indexof the frame look_back_time amount from when the food is brought to the mouth
    # if look back time is greater than number of frames availible it looks back as far as possible
    index_look_back = closest(img_float_list, (mouth_to_food - look_back_time))
    
    # gets the frames we want to make a video of and its fps
    frames_to_extract = img_path_list[index_look_back[0]: index_food_touched[0]]
    floats_frames_to_extract = img_float_list[index_look_back[0]: index_food_touched[0]]
    vid_fps = get_fps(floats_frames_to_extract)
    
    # makes the extracted video`
    if os.path.exists(vid_name) != True:
        make_video(frames_to_extract,
                   vid_name,
                   vid_fps)
    else:
        print(vid_name + ' already exists!')
        
    dat_extracted_frames = pd.DataFrame(list(zip(frames_to_extract,
                                        floats_frames_to_extract)),
                                        columns = ['frame_paths', 'frame_floats'])
    if os.path.exists(csv_vid_name) != True:
        dat_extracted_frames.to_csv(csv_vid_name)
    else:
        print(csv_vid_name + ' already exists')
    
    print()
    
    
    
    
    
    
    
    
        
    