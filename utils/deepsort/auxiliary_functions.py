import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
# utils
from utils import score
# deepsort
from deepsort.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from deepsort.deep_sort_pytorch.utils.parser import get_config
# deepsort utils
from utils.deepsort import plot_boxes as pb

def deepsort_helmets(video, video_data, video_dir, 
    deepsort_config='deepsort/config.yaml',
    plot=False, plot_frames=[]
):
    
    # Setup Deepsort
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)    
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, #nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    
    # Run through frames.
    video_data = video_data.sort_values('frame').reset_index(drop=True)
    ds = []
    for frame, d in tqdm(video_data.groupby(['frame']), total=video_data['frame'].nunique()):
        d['x'] = (d['left'] + round(d['width'] / 2))
        d['y'] = (d['top'] + round(d['height'] / 2))

        xywhs = d[['x','y','width','height']].values

        cap = cv2.VideoCapture(f'{video_dir}/{video}.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1) # optional
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        confs = np.ones([len(d),])
        clss =  np.zeros([len(d),])
        outputs = deepsort.update(xywhs, confs, clss, image)

        if (plot and frame > cfg.DEEPSORT.N_INIT) or (frame in plot_frames):
            for j, (output, conf) in enumerate(zip(outputs, confs)): 

                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id}'
                color = pb.compute_color_for_id(id)
                im = pb.plot_one_box(bboxes, image, label=label, color=color, line_thickness=2)
            fig, ax = plt.subplots(figsize=(15, 10))
            video_frame = d['video_frame'].values[0]
            ax.set_title(f'Deepsort labels: {video_frame}')
            plt.imshow(im)
            plt.show()

        preds_df = pd.DataFrame(outputs, columns=['left','top','right','bottom','deepsort_cluster','class'])
        if len(preds_df) > 0:
            # TODO Fix this messy merge
            d = d.astype({"left":np.int32, "width":np.int32, "top":np.int32, "height":np.int32})
            d = pd.merge_asof(d.sort_values(['left','top']),
                              preds_df[['left','top','deepsort_cluster']] \
                              .sort_values(['left','top']), on='left', suffixes=('','_deepsort'),
                              direction='nearest')
        ds.append(d)
    dout = pd.concat(ds)
    return dout

def add_deepsort_label_col(out):
    # Find the top occuring label for each deepsort_cluster
    sortlabel_map = out.groupby('deepsort_cluster')['label'].value_counts() \
        .sort_values(ascending=False).to_frame() \
        .rename(columns={'label':'label_count'}) \
        .reset_index() \
        .groupby(['deepsort_cluster']) \
        .first()['label'].to_dict()
    # Find the # of times that label appears for the deepsort_cluster.
    sortlabelcount_map = out.groupby('deepsort_cluster')['label'].value_counts() \
        .sort_values(ascending=False).to_frame() \
        .rename(columns={'label':'label_count'}) \
        .reset_index() \
        .groupby(['deepsort_cluster']) \
        .first()['label_count'].to_dict()
    
    out['label_deepsort'] = out['deepsort_cluster'].map(sortlabel_map)
    out['label_count_deepsort'] = out['deepsort_cluster'].map(sortlabelcount_map)

    return out

def score_vs_deepsort(myvideo, out, labels):
    # Score the base predictions compared to the deepsort postprocessed predictions.
    myvideo_mp4 = myvideo + '.mp4'
    labels_video = labels.query('video == @myvideo_mp4')
    scorer = score.NFLAssignmentScorer(labels_video)
    out_deduped = out.groupby(['video_frame','label']).first().reset_index()
    base_video_score = scorer.score(out_deduped)
    
    out_preds = out.drop('label', axis=1).rename(columns={'label_deepsort':'label'})
    out_preds = out_preds.groupby(['video_frame','label']).first().reset_index()
    deepsort_video_score = scorer.score(out_preds)
    print(f'{base_video_score:0.5f} before --> {deepsort_video_score:0.5f} deepsort')

    return base_video_score, deepsort_video_score


def submission_deepsort(base_submission, video_dir, debug, labels=pd.DataFrame()):
    # Add video and frame columns to submission.
    base_submission['video'] = base_submission['video_frame'].str.split('_').str[:3].str.join('_')
    base_submission['frame'] = base_submission['video_frame'].str.split('_').str[-1].astype('int')

    # Loop through test videos and apply. If in debug mode show the score change.
    out_ds = []
    outs = []
    for myvideo, video_data in tqdm(base_submission.groupby('video'), total=base_submission['video'].nunique()):
        print(f'==== {myvideo} ====')
        if debug:
            # Plot deepsort labels when in debug mode.
            out = deepsort_helmets(myvideo, video_data, video_dir, plot_frames=[10, 150, 250])
        else:
            out = deepsort_helmets(myvideo, video_data, video_dir)
        out_ds.append(out)
        out = add_deepsort_label_col(out)
        outs.append(out)
        if debug and not labels.empty:
            # Score
            score_vs_deepsort(myvideo, out, labels)

    submission_deepsort = pd.concat(outs).copy()
    return submission_deepsort