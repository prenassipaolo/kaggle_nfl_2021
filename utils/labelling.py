import numpy as np
import pandas as pd
from tqdm import tqdm
import time

def random_submission(helmets, tracks):
    """
    Creates a baseline submission with randomly assigned helmets
    based on the top 22 most confident baseline helmet boxes for
    a frame.
    """
    # Take up to 22 helmets per frame based on confidence:
    helm_22 = (
        helmets.sort_values("conf", ascending=False)
        .groupby("video_frame")
        .head(22)
        .sort_values("video_frame")
        .reset_index(drop=True)
        .copy()
    )
    # Identify player label choices for each game_play
    game_play_choices = tracks.groupby(["game_play"])["player"].unique().to_dict()
    # Loop through frames and randomly assign boxes
    ds = []
    helm_22["label"] = np.nan
    i=0
    # assigning helmets to players
    print('Assigning helmets to players...')
    for video_frame, data in tqdm(helm_22.groupby("video_frame")):
        i+=1
        game_play = video_frame[:12]
        choices = game_play_choices[game_play]
        np.random.shuffle(choices)
        data["label"] = choices[: len(data)]
        ds.append(data)
    # concatenate dataframes and print the time
    print('Concatenating...\t')
    tt = time.time()
    submission = pd.concat(ds)
    print(f'Time: {int(time.time()-tt)} seconds')
    
    return submission