import os
import os.path as osp
import json
from tqdm import tqdm
import cv2

def split_video_into_shots(imgs_folder, video_path, boundaries):
    video_name = video_path.split('_')[0]
    video_name = video_name.split('/')[-1]
    print(video_name)
    cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    crop_params = {'left': 255, 'top': 120, 'width': 1050, 'height': 790}
    for boundary in boundaries:
        shot_id = boundary['shotId']
        in_point = boundary['inPoint']
        out_point = boundary['outPoint']
        shot_type = boundary['shotType']
        if shot_type in ['T', 'NA'] or out_point - in_point < 21:
            continue
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, in_point)
        for i in range(in_point, out_point - 9):
            ret, frame = cap.read()
            if i % 10 != 0:
                continue
            if not ret:
                break
            frame = frame[crop_params['top']:crop_params['top']+crop_params['height'], 
                          crop_params['left']:crop_params['left']+crop_params['width']]
            # print(frame.shape)
            if frame is not None:
                if frame.shape[0] != 0 and frame.shape[1] != 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, (256, 256))
                    frames.append(frame)
        if len(frames)>1:
            for fi in range(len(frames)):
                img_path = osp.join(imgs_folder, video_name + '_' + str(shot_id) + '_' + str(fi)+'.jpg')
                # print(img_path)
                cv2.imwrite(img_path, frames[fi])
    cap.release()

def get_shot_frames(path):
    videos_folder = osp.join(path, 'videos')
    imgs_folder = osp.join(path, 'val/imgs')
    boundarys_folder = osp.join(path, 'boundarys')
    video_names_sorted = sorted(os.listdir(videos_folder), key=lambda k: float(k.split("_")[0]))
    video_paths = [os.path.join(videos_folder, video_file) for video_file in video_names_sorted]

    for video_path in tqdm(video_paths):
        video_name = osp.splitext(osp.basename(video_path))[0]
        boundary_path = osp.join(boundarys_folder, video_name.split("_")[0] + '-shot_annotations.json')

        with open(boundary_path, 'r') as f:
            boundaries = json.load(f)
        
        split_video_into_shots(imgs_folder, video_path, boundaries)


path = r'./data/'
get_shot_frames(path)