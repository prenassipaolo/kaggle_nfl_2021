import numpy as np
import pandas as pd
import itertools
import glob
import os
import cv2
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import random

def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = tracks["gameKey"].astype("str") + "_" + tracks["playID"].astype("str").str.zfill(6)
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = tracks.query('event == "ball_snap"').groupby("game_play")["time"].first().to_dict()
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype("timedelta64[ms]") / 1_000
    # Estimated video frame
    tracks["est_frame"] = ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    return tracks


    # DATA LOAD
BASE_DIR = 'data/nfl-health-and-safety-helmet-assignment/'

n_test_videos = len(os.listdir(BASE_DIR+'test/'))
# Run in debug mode unless during submission
if n_test_videos == 6:
    debug = True
else:
    debug = False

# Configurables
n_debug_samples = 1
random_state = 2020
CONF_THRE = 0.3
max_iter = 1000
DIG_STEP = 3
DIG_MAX = DIG_STEP*10

# Read in the data.


labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
if debug:
    tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
    helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
else:
    tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')
    helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')
    
tracking = add_track_features(tracking)


# cambio label delle colonne, separa game key nei vari elementi
def add_cols(df):
    df['game_play'] = df['video_frame'].str.split('_').str[:2].str.join('_')
    if 'video' not in df.columns:
        df['video'] = df['video_frame'].str.split('_').str[:3].str.join('_') + '.mp4'
    return df

if debug:
    helmets = add_cols(helmets)
    labels = add_cols(labels)
    # Select `n_debug_samples` worth of videos to debug with
    sample_videos = labels['video'].drop_duplicates() \
        .sample(n_debug_samples, random_state=random_state).tolist()
    sample_gameplays = ['_'.join(x.split('_')[:2]) for x in sample_videos]
    tracking = tracking[tracking['game_play'].isin(sample_gameplays)]
    helmets = helmets[helmets['video'].isin(sample_videos)]
    labels = labels[labels['video'].isin(sample_videos)]
tracking.shape, helmets.shape, labels.shape

def find_nearest(array, value):
    value = int(value)
    array = np.asarray(array).astype(int)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def norm_arr(a):
    a = a-a.min()
    a = a/a.max()
    return a
    
def dist(a1, a2):
    return np.linalg.norm(a1-a2)

def dist_for_different_len(a1, a2):
    '''deletes players that do not appear in the frame (iterative random process)
    '''
    assert len(a1) >= len(a2), f'{len(a1)}, {len(a2)}'
    len_diff = len(a1) - len(a2)
    a2 = norm_arr(a2)
    if len_diff == 0:
        a1 = norm_arr(a1)
        return dist(a1,a2), ()
    else:
        min_dist = 10000
        min_detete_idx = None
        cnt = 0
        del_list = list(itertools.combinations(range(len(a1)),len_diff))
        if len(del_list) > max_iter:
            del_list = random.sample(del_list, max_iter)
        for detete_idx in del_list:
            this_a1 = np.delete(a1, detete_idx)
            this_a1 = norm_arr(this_a1)
            this_dist = dist(this_a1, a2)
            #print(len(a1), len(a2), this_dist)
            if min_dist > this_dist:
                min_dist = this_dist
                min_detete_idx = detete_idx
                
        return min_dist, min_detete_idx
        
def rotate_arr(u, t, deg=True):
    if deg == True:
        t = np.deg2rad(t)
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return  np.dot(R, u)

def dist_rot(tracking_df, a2): 
    '''finds the angle with minimal distance by brute force
    '''
    tracking_df = tracking_df.sort_values('x')
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    for dig in range(-DIG_MAX,DIG_MAX+1,DIG_STEP):
        arr = rotate_arr(np.array((x,y)), dig)
        this_dist, this_idx = dist_for_different_len(np.sort(arr[0]), a2)
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_x = arr[0]
    tracking_df['x_rot'] = min_x
    player_arr = tracking_df.sort_values('x_rot')['player'].values
    players = np.delete(player_arr,min_idx)
    return min_dist, players


def mapping_df(args):
    video_frame, df = args
    gameKey,playID,view,frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)
    this_tracking = tracking[(tracking['gameKey']==gameKey) & (tracking['playID']==playID)]
    est_frame = find_nearest(this_tracking.est_frame.values, frame)
    this_tracking = this_tracking[this_tracking['est_frame']==est_frame]
    len_this_tracking = len(this_tracking)
    df['center_h_p'] = (df['left']+df['width']/2).astype(int)
    df['center_h_m'] = (df['left']+df['width']/2).astype(int)*-1
    df = df[df['conf']>CONF_THRE].copy()
    if len(df) > len_this_tracking:
        df = df.tail(len_this_tracking)
    df_p = df.sort_values('center_h_p').copy()
    df_m = df.sort_values('center_h_m').copy()
    
    if view == 'Endzone':
        this_tracking['x'], this_tracking['y'] = this_tracking['y'].copy(), this_tracking['x'].copy()
    a2_p = df_p['center_h_p'].values
    a2_m = df_m['center_h_m'].values

    min_dist_p, min_detete_idx_p = dist_rot(this_tracking ,a2_p)
    min_dist_m, min_detete_idx_m = dist_rot(this_tracking ,a2_m)
    if min_dist_p < min_dist_m:
        min_dist = min_dist_p
        min_detete_idx = min_detete_idx_p
        tgt_df = df_p
    else:
        min_dist = min_dist_m
        min_detete_idx = min_detete_idx_m
        tgt_df = df_m
    #print(video_frame, len(this_tracking), len(df), len(df[df['conf']>CONF_THRE]), this_tracking['x'].mean(), min_dist_p, min_dist_m, min_dist)
    tgt_df['label'] = min_detete_idx
    return tgt_df[['video_frame','left','width','top','height','label']]

#p = Pool(processes=4)
submission_df_list = []
df_list = list(helmets.groupby('video_frame'))
with tqdm(total=len(df_list)) as pbar:
    #for this_df in p.imap(mapping_df, df_list):
    for this_df in map(mapping_df, df_list):
        submission_df_list.append(this_df)
        pbar.update(1)
p.close()

submission_df = pd.concat(submission_df_list)
submission_df.to_csv('submission-baseline.csv', index=False)