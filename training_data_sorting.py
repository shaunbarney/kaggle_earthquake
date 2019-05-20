"""Script to sort training data to match test data"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CWD = os.getcwd()
DATASET_DIR = os.path.join(CWD, 'dataset')


def check_test_data():
    fig, axes = plt.subplots(figsize=(12, 8))
    test_dir = os.path.join(DATASET_DIR, 'test')
    test_seg = os.listdir(test_dir)[120]
    seg_df = pd.read_csv(os.path.join(test_dir, test_seg), dtype={'acoustic_data': np.int16})
    print(seg_df.head())
    print(len(seg_df))
    plt.plot(seg_df.acoustic_data)
    plt.show()


def create_training_set():
    for i in range(4194):
        print(f"i: {i}")
        df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'),
                         skiprows=i,
                         nrows=150000,
                         dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},
                         names=[0, 1])
        df.to_csv(os.path.join(DATASET_DIR, 'training_segments', f"seg_{i}.csv"))


# 150000 samples in test
# 629145480 total in training
# 4194 training segments
if __name__ == '__main__':
    # check_test_data()
    create_training_set()
