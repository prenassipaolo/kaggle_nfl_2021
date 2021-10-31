import pandas as pd

def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    # convert the time type from string to datetime
    tracks["time"] = pd.to_datetime(tracks["time"])
    # identify ball snaps to syncing the videos according to the data descrtiption
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    # associate the snap time to each play
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    # flag each frame as snap or not
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    # calculate in milliseconds the difference 
    # between the snap time and the actual frame time and divide by 1000
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
        "timedelta64[ms]"
    ) / 1_000
    # calculate the estimated video frame unit according to the video
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks


def add_cols(df):
    """
    Split the informations of the 'video_frame' colums
    """
    df['game_play'] = df['video_frame'].str.split('_').str[:2].str.join('_')
    if 'video' not in df.columns:
        df['video'] = df['video_frame'].str.split('_').str[:3].str.join('_') + '.mp4'
    return df