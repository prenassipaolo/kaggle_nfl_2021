import os
import cv2
import pandas as pd
import matplotlib.pylab as plt


def video_with_baseline_boxes(
    video_path: str, baseline_boxes: pd.DataFrame, gt_labels: pd.DataFrame, verbose=True
) -> str:
    """
    Annotates a video with both the baseline model boxes and ground truth boxes.
    Baseline model prediction confidence is also displayed.

    Possible incompatibilities with web browsers due to video format (codec)
    """
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)  # Black
    BASELINE_COLOR = (255, 255, 255)  # White
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy()
    gt_labels = gt_labels.copy()

    baseline_boxes["video"] = (
        baseline_boxes["video_frame"].str.split("_").str[:3].str.join("_")
    )
    gt_labels["video"] = gt_labels["video_frame"].str.split("_").str[:3].str.join("_")
    baseline_boxes["frame"] = (
        baseline_boxes["video_frame"].str.split("_").str[-1].astype("int")
    )
    gt_labels["frame"] = gt_labels["video_frame"].str.split("_").str[-1].astype("int")

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name + ".mp4"
    
    # delete previuous files with the same name
    if os.path.exists(output_path):
        os.remove(output_path)
    
    output_video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"
        cv2.putText(
            img,
            img_name,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            HELMET_COLOR,
            thickness=2,
        )

        # Now, add the boxes
        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                BASELINE_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                f"{box.conf:0.2}",
                (box.left, max(0, box.top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BASELINE_COLOR,
                thickness=1,
            )

        boxes = gt_labels.query("video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            # Filter for definitive head impacts and turn labels red
            if box.isDefinitiveImpact == True:
                color, thickness = IMPACT_COLOR, 3
            else:
                color, thickness = HELMET_COLOR, 1
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                color,
                thickness=thickness,
            )
            cv2.putText(
                img,
                box.label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
            )

        output_video.write(img)
    output_video.release()

    return output_path


def video(path):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame', frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        # Break the loop
        else: 
            break
    
    # When everything done, release 
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    return


def frame(video_path, frame, event=None):
    """
    Plot the image of the specified video at the specified frame
    """
    # select the video
    cap = cv2.VideoCapture(video_path)
    # set the correct frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    # read the data
    _, ez_snap_img = cap.read()
    
    # plot the frame
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(ez_snap_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    title = f"{video_path.split('/')[-1]}"
    if event:
        title += f" ({event})"
    plt.title(title, fontsize=20)

    return


