import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Read in data files
BASE_DIR = 'data/nfl-health-and-safety-helmet-assignment'
#on kaggle: '../input/nfl-health-and-safety-helmet-assignment'

# Labels and sample submission
labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')

# Player tracking data
tr_tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
te_tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')

# Baseline helmet detection labels
tr_helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
te_helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')

# Extra image labels
img_labels = pd.read_csv(f'{BASE_DIR}/image_labels.csv')

print(ss.head())