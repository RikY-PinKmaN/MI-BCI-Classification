# wgan_gp_project/main.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.signal import ellip, ellipord, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import os
import time
import gc
import copy
import traceback
import warnings
import scipy.io
import pandas as pd
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# --- Configuration ---
NUM_MAIN_ITERATIONS = 1  # Number of full runs for reproducibility
LOWCUT = 8               # Lower frequency bound (Hz)
HIGHCUT = 35             # Upper frequency bound (Hz)
FS = 250                 # Sampling frequency (Hz)

# GAN parameters
Z_DIM = 100              # Dimension of the noise vector
GAN_EPOCHS = 2000       # Epochs for WGAN training
BATCH_SIZE = 10          # Batch size for GAN training
CSP_COUNT = 3            # Number of CSP components
CSP_REG = 0.1            # Regularization for CSP
LAMBDA_GP = 10           # Gradient penalty coefficient for WGAN-GP
LAMBDA_CLS = 1.5        # Classification guidance strength
CRITIC_ITERATIONS = 3    # Number of critic updates per generator update
MIX_RATIOS = [25, 50, 100, 200]  # Percentages of synthetic data to mix
        # Random seed for reproducibility

# Preprocessing & Split Parameters
N_TRAIN_PER_CLASS = 50   # Trials per class for Training set
N_VAL_PER_CLASS = 10      # Trials per class for Validation set
NUM_GAN_RUNS = 3         # Number of GAN training runs
NUM_BATCHES_PER_RUN = 20 # Number of batches to generate per GAN

# Placeholders
N_CHANNELS = 22          # Number of EEG channels
N_TIMEPOINTS = 500       # Expected timepoints after preprocessing

# Output Directory Setup
BASE_OUTPUT_DIR = "WGAN_GP_CL_S2-1_ftr_per_class"  # Directory for results
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Common Preprocessing Function: Elliptical Filter ---
def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    nyq = 0.5 * fs
    wp = [lowcut / nyq, highcut / nyq]  # Passband
    ws = [(lowcut - 1) / nyq, (highcut + 1) / nyq]  # Stopband
    n, wn = ellipord(wp, ws, rp, rs)  # Filter order and natural frequency
    b, a = ellip(n, rp, rs, wn, btype='band')  # Coefficients
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def _normalize_trials_internal(data):
    """
    Placeholder for normalizing trials to the range [-1, 1].
    Args:
        data (np.ndarray): Data to normalize (trials, channels, time_points).
    Returns:
        np.ndarray: Normalized data.
    """
    normalized_data = np.empty_like(data)
    if data.shape[0] == 0: # Handle empty data case (no trials)
        return normalized_data
        
    for i in range(data.shape[0]): # Iterate over trials
        trial = data[i, :, :] # trial is (channels, time_points)
        min_val = trial.min()
        max_val = trial.max()
        if max_val - min_val > 1e-6: # Avoid division by zero or near-zero
            normalized_data[i, :, :] = 2 * (trial - min_val) / (max_val - min_val) - 1
        else:
            # If min and max are the same, trial is constant.
            # Normalize to 0 if it's all same, or keep as is if it's already within [-1,1]
            # For simplicity, just copy if range is too small.
            normalized_data[i, :, :] = trial 
    return normalized_data

# # --- Main Combined Preprocessing Function ---
# def eeg_preprocess(
#     x_data,
#     y_data,
#     subject_id=0,
#     train_trials_per_class=N_TRAIN_PER_CLASS,
#     val_trials_per_class=N_VAL_PER_CLASS,
#     lowcut=LOWCUT,
#     highcut=HIGHCUT,
#     fs=FS,
#     time_window_start=115,
#     time_window_end_exclusive=615,
#     label_left=1,
#     label_right=2
# ):
#     """
#     Combines preprocessing logic for EEG data, including random splitting,
#     filtering, time windowing, transposing, and normalization.

#     Args:
#         x_data (np.ndarray): Raw EEG data with shape (time_points, channels, total_trials).
#         y_data (np.ndarray): Labels for each trial, shape (total_trials,).
#                              Assumes labels for left and right motor imagery.
#         subject_id (int, optional): Identifier for the subject (for logging). Defaults to 0.
#         train_trials_per_class (int, optional): Number of trials per class for training. Defaults to 50.
#         val_trials_per_class (int, optional): Number of trials per class for validation. Defaults to 10.
#         lowcut (float, optional): Lower cutoff frequency for bandpass filter. Defaults to 8.
#         highcut (float, optional): Higher cutoff frequency for bandpass filter. Defaults to 35.
#         fs (float, optional): Sampling frequency of the data. Defaults to 250.
#         time_window_start (int, optional): Start index for the time window. Defaults to 115.
#         time_window_end_exclusive (int, optional): End index (exclusive) for the time window.
#                                                  Defaults to 615 (results in 500 time points if start is 115).
#         label_left (int, optional): Label used for 'left' class. Defaults to 1.
#         label_right (int, optional): Label used for 'right' class. Defaults to 2.

#     Returns:
#         tuple: Contains the following elements:
#             - x_train_final (np.ndarray): Processed training data (trials, channels, time_points_windowed).
#             - y_train (np.ndarray): Training labels (trials,).
#             - x_val_final (np.ndarray): Processed validation data.
#             - y_val (np.ndarray): Validation labels.
#             - x_test_final (np.ndarray): Processed test data.
#             - y_test (np.ndarray): Test labels.
#             - split_indices (dict): Dictionary containing the indices used for splitting, relative
#                                     to the original per-class trial arrays. Keys:
#                                     'train_right_idx', 'train_left_idx',
#                                     'val_right_idx', 'val_left_idx',
#                                     'test_right_idx', 'test_left_idx'.
#             - finaltrn (dict): Dictionary for training data {'x': (time_points_windowed, channels, trials), 'y': labels}.
#             - finalval (dict): Dictionary for validation data {'x': (time_points_windowed, channels, trials), 'y': labels}.
#             - finaltest (dict): Dictionary for test data {'x': (time_points_windowed, channels, trials), 'y': labels, 'lowfreq', 'highfreq'}.
#             Returns (None, None, None, None, None, None, None, None, None, None) on error or insufficient data.
#     """
#     try:
#         print(f"--- Starting preprocessing for Subject S{subject_id} ---")
#         y_data_flat = y_data.ravel()

#         data_right_all = x_data[:, :, y_data_flat == label_right]
#         data_left_all = x_data[:, :, y_data_flat == label_left]

#         n_trials_right_total = data_right_all.shape[2]
#         n_trials_left_total = data_left_all.shape[2]
#         print(f"  Available trials: Right={n_trials_right_total}, Left={n_trials_left_total}")

#         min_required_trials = train_trials_per_class + val_trials_per_class
#         if n_trials_right_total < min_required_trials or n_trials_left_total < min_required_trials: # Strict check for train+val
#             print(f"  WARNING S{subject_id}: Not enough trials (R:{n_trials_right_total}, L:{n_trials_left_total}) "
#                   f"to select {train_trials_per_class} train + {val_trials_per_class} val per class. Skipping.")
#             return None, None, None, None, None, None, None, None, None, None

#         all_right_indices = np.arange(n_trials_right_total)
#         all_left_indices = np.arange(n_trials_left_total)

#         selected_train_right_idx = np.random.choice(all_right_indices, train_trials_per_class, replace=False)
#         selected_train_left_idx = np.random.choice(all_left_indices, train_trials_per_class, replace=False)

#         remaining_right_indices = np.setdiff1d(all_right_indices, selected_train_right_idx, assume_unique=True)
#         remaining_left_indices = np.setdiff1d(all_left_indices, selected_train_left_idx, assume_unique=True)

#         if len(remaining_right_indices) < val_trials_per_class or len(remaining_left_indices) < val_trials_per_class:
#             print(f"  WARNING S{subject_id}: Not enough remaining trials after training selection "
#                   f"(R:{len(remaining_right_indices)}, L:{len(remaining_left_indices)}) "
#                   f"to select {val_trials_per_class} validation trials per class. Skipping.")
#             return None, None, None, None, None, None, None, None, None, None
            
#         selected_val_right_idx = np.random.choice(remaining_right_indices, val_trials_per_class, replace=False)
#         selected_val_left_idx = np.random.choice(remaining_left_indices, val_trials_per_class, replace=False)

#         test_right_idx = np.setdiff1d(remaining_right_indices, selected_val_right_idx, assume_unique=True)
#         test_left_idx = np.setdiff1d(remaining_left_indices, selected_val_left_idx, assume_unique=True)
        
#         split_indices = {
#             'train_right_idx': selected_train_right_idx, 'train_left_idx': selected_train_left_idx,
#             'val_right_idx': selected_val_right_idx, 'val_left_idx': selected_val_left_idx,
#             'test_right_idx': test_right_idx, 'test_left_idx': test_left_idx
#         }
#         print(f"  Split indices generated: "
#               f"Train (R:{len(selected_train_right_idx)}, L:{len(selected_train_left_idx)}), "
#               f"Val (R:{len(selected_val_right_idx)}, L:{len(selected_val_left_idx)}), "
#               f"Test (R:{len(test_right_idx)}, L:{len(test_left_idx)})")

#         train_x_right = data_right_all[:, :, selected_train_right_idx]
#         train_x_left = data_left_all[:, :, selected_train_left_idx]
#         val_x_right = data_right_all[:, :, selected_val_right_idx]
#         val_x_left = data_left_all[:, :, selected_val_left_idx]

#         time_points_dim = data_right_all.shape[0]
#         channels_dim = data_right_all.shape[1]
#         windowed_time_points_count = time_window_end_exclusive - time_window_start

#         test_x_right = data_right_all[:, :, test_right_idx] if len(test_right_idx) > 0 else np.empty((time_points_dim, channels_dim, 0))
#         test_x_left = data_left_all[:, :, test_left_idx] if len(test_left_idx) > 0 else np.empty((time_points_dim, channels_dim, 0))

#         print(f"  Applying time window: {time_window_start} to {time_window_end_exclusive-1}")
#         train_x_right_win = train_x_right[time_window_start:time_window_end_exclusive, :, :]
#         train_x_left_win  = train_x_left[time_window_start:time_window_end_exclusive, :, :]
#         val_x_right_win   = val_x_right[time_window_start:time_window_end_exclusive, :, :]
#         val_x_left_win    = val_x_left[time_window_start:time_window_end_exclusive, :, :]
        
#         test_x_right_win = test_x_right[time_window_start:time_window_end_exclusive, :, :] if test_x_right.shape[2] > 0 else np.empty((windowed_time_points_count, channels_dim, 0))
#         test_x_left_win = test_x_left[time_window_start:time_window_end_exclusive, :, :] if test_x_left.shape[2] > 0 else np.empty((windowed_time_points_count, channels_dim, 0))

#         x_train_cat = np.concatenate([train_x_right_win, train_x_left_win], axis=2)
#         x_val_cat   = np.concatenate([val_x_right_win, val_x_left_win], axis=2)
#         x_test_cat  = np.concatenate([test_x_right_win, test_x_left_win], axis=2)

#         y_train = np.concatenate([np.full(train_x_right_win.shape[2], label_right), np.full(train_x_left_win.shape[2], label_left)])
#         y_val   = np.concatenate([np.full(val_x_right_win.shape[2], label_right), np.full(val_x_left_win.shape[2], label_left)])
#         y_test  = np.concatenate([np.full(test_x_right_win.shape[2], label_right), np.full(test_x_left_win.shape[2], label_left)])
        
#         print(f"  Concatenated data shapes before filtering (Time,Chans,Trials): Train X={x_train_cat.shape}, Y={y_train.shape}")

#         print(f"  Applying elliptical filter (lowcut={lowcut}, highcut={highcut}, fs={fs})...")
#         filtered_datasets_trials_first = []
#         for data_cat in [x_train_cat, x_val_cat, x_test_cat]:
#             if data_cat.shape[2] == 0: # No trials in this set
#                 filtered_datasets_trials_first.append(np.empty((0, windowed_time_points_count, channels_dim)))
#                 continue
#             # Transpose to (trials, time_points_windowed, channels) for iterating trials
#             data_trials_first = np.transpose(data_cat, (2, 0, 1))
#             processed_trials_list = [elliptical_filter(trial_data, lowcut, highcut, fs) for trial_data in data_trials_first]
#             filtered_datasets_trials_first.append(np.array(processed_trials_list))

#         x_train_filtered_tf, x_val_filtered_tf, x_test_filtered_tf = filtered_datasets_trials_first
#         print(f"  Filtered data shapes (Trials,Time,Chans): Train={x_train_filtered_tf.shape}")

#         # Transpose to Final X Shape for WGAN/CSP: (trials, channels, time_points_windowed)
#         x_train_processed = np.transpose(x_train_filtered_tf, (0, 2, 1)) if x_train_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
#         x_val_processed   = np.transpose(x_val_filtered_tf, (0, 2, 1)) if x_val_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
#         x_test_processed  = np.transpose(x_test_filtered_tf, (0, 2, 1)) if x_test_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
#         print(f"  Transposed data shapes for normalization (Trials,Chans,Time): Train={x_train_processed.shape}")

#         print("  Normalizing data to [-1, 1]...")
#         x_train_final = _normalize_trials_internal(x_train_processed)
#         x_val_final   = _normalize_trials_internal(x_val_processed)
#         x_test_final  = _normalize_trials_internal(x_test_processed)
#         print(f"  Final data shapes (x_..._final - Trials,Chans,Time): Train X={x_train_final.shape}")

#         # --- Create finaltrn, finalval, finaltest dictionaries ---
#         # These will have 'x' data in shape (time_points_windowed, channels, trials)
#         # The data is already filtered and normalized from x_train_final etc.
#         finaltrn = {
#             'x': np.transpose(x_train_final, (2, 1, 0)) if x_train_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
#             'y': y_train
#         }
#         finalval = {
#             'x': np.transpose(x_val_final, (2, 1, 0)) if x_val_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
#             'y': y_val
#         }
#         finaltest = {
#             'x': np.transpose(x_test_final, (2, 1, 0)) if x_test_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
#             'y': y_test,
#             'lowfreq': lowcut,
#             'highfreq': highcut
#         }
#         print(f"  finaltrn shapes: x={finaltrn['x'].shape}, y={finaltrn['y'].shape}")
#         print(f"--- Preprocessing for Subject S{subject_id} COMPLETED ---")

#         return x_train_final, y_train, finaltrn, finalval, finaltest

#     except Exception as e:
#         print(f"  ERROR S{subject_id} during combined preprocessing: {e}")
#         traceback.print_exc()
#         return None, None, None, None, None, None, None, None, None, None
    
def eeg_preprocess(
    x_data,
    y_data,
    tx_data,
    ty_data,
    subject_id=0,
    train_trials_per_class=None,  # Now optional since we use all training data
    val_trials_per_class=N_VAL_PER_CLASS,
    lowcut=LOWCUT,
    highcut=HIGHCUT,
    fs=FS,
    time_window_start=115,
    time_window_end_exclusive=615,
    label_left=1,
    label_right=2
):
    """
    Preprocesses EEG data using separate data sources for training and validation/testing.
    
    Args:
        x_data (np.ndarray): Training EEG data with shape (time_points, channels, total_trials).
        y_data (np.ndarray): Training labels for each trial, shape (total_trials,).
        tx_data (np.ndarray): Validation/Testing EEG data with shape (time_points, channels, total_trials).
        ty_data (np.ndarray): Validation/Testing labels for each trial, shape (total_trials,).
        subject_id (int, optional): Identifier for the subject (for logging). Defaults to 0.
        train_trials_per_class (int, optional): If specified, limits number of trials per class for training.
                                               If None, uses all available training data. Defaults to None.
        val_trials_per_class (int, optional): Number of trials per class for validation. Defaults to 10.
        lowcut (float, optional): Lower cutoff frequency for bandpass filter. Defaults to 8.
        highcut (float, optional): Higher cutoff frequency for bandpass filter. Defaults to 35.
        fs (float, optional): Sampling frequency of the data. Defaults to 250.
        time_window_start (int, optional): Start index for the time window. Defaults to 115.
        time_window_end_exclusive (int, optional): End index (exclusive) for the time window.
                                                 Defaults to 615 (results in 500 time points if start is 115).
        label_left (int, optional): Label used for 'left' class. Defaults to 1.
        label_right (int, optional): Label used for 'right' class. Defaults to 2.

    Returns:
        tuple: Contains the following elements:
            - x_train_final (np.ndarray): Processed training data (trials, channels, time_points_windowed).
            - y_train (np.ndarray): Training labels (trials,).
            - x_val_final (np.ndarray): Processed validation data.
            - y_val (np.ndarray): Validation labels.
            - x_test_final (np.ndarray): Processed test data.
            - y_test (np.ndarray): Test labels.
            - split_indices (dict): Dictionary containing the indices used for splitting.
            - finaltrn (dict): Dictionary for training data {'x': (time_points_windowed, channels, trials), 'y': labels}.
            - finalval (dict): Dictionary for validation data {'x': (time_points_windowed, channels, trials), 'y': labels}.
            - finaltest (dict): Dictionary for test data {'x': (time_points_windowed, channels, trials), 'y': labels, 'lowfreq', 'highfreq'}.
    """
    try:
        print(f"--- Starting preprocessing for Subject S{subject_id} ---")
        
        # Process training data
        y_data_flat = y_data.ravel()
        data_right_train = x_data[:, :, y_data_flat == label_right]
        data_left_train = x_data[:, :, y_data_flat == label_left]

        n_trials_right_train = data_right_train.shape[2]
        n_trials_left_train = data_left_train.shape[2]
        print(f"  Available training trials: Right={n_trials_right_train}, Left={n_trials_left_train}")

        # Process validation/test data
        ty_data_flat = ty_data.ravel()
        data_right_valtest = tx_data[:, :, ty_data_flat == label_right]
        data_left_valtest = tx_data[:, :, ty_data_flat == label_left]

        n_trials_right_valtest = data_right_valtest.shape[2]
        n_trials_left_valtest = data_left_valtest.shape[2]
        print(f"  Available validation/test trials: Right={n_trials_right_valtest}, Left={n_trials_left_valtest}")

        # Check if we have enough trials for validation
        if n_trials_right_valtest < val_trials_per_class or n_trials_left_valtest < val_trials_per_class:
            print(f"  WARNING S{subject_id}: Not enough trials in validation/test set "
                  f"(R:{n_trials_right_valtest}, L:{n_trials_left_valtest}) "
                  f"to select {val_trials_per_class} validation trials per class. Skipping.")
            return None, None, None, None, None, None, None, None, None, None

        # Handle training set selection
        if train_trials_per_class is not None and (n_trials_right_train < train_trials_per_class or n_trials_left_train < train_trials_per_class):
            print(f"  WARNING S{subject_id}: Not enough trials in training set "
                  f"(R:{n_trials_right_train}, L:{n_trials_left_train}) "
                  f"to select {train_trials_per_class} training trials per class. Skipping.")
            return None, None, None, None, None, None, None, None, None, None

        # Select training data (all or subset)
        all_right_train_indices = np.arange(n_trials_right_train)
        all_left_train_indices = np.arange(n_trials_left_train)
        
        if train_trials_per_class is not None:
            selected_train_right_idx = np.random.choice(all_right_train_indices, train_trials_per_class, replace=False)
            selected_train_left_idx = np.random.choice(all_left_train_indices, train_trials_per_class, replace=False)
        else:
            # Use all training data
            selected_train_right_idx = all_right_train_indices
            selected_train_left_idx = all_left_train_indices

        # Select validation data
        all_right_valtest_indices = np.arange(n_trials_right_valtest)
        all_left_valtest_indices = np.arange(n_trials_left_valtest)
        
        selected_val_right_idx = np.random.choice(all_right_valtest_indices, val_trials_per_class, replace=False)
        selected_val_left_idx = np.random.choice(all_left_valtest_indices, val_trials_per_class, replace=False)
        
        # Remaining are for testing
        test_right_idx = np.setdiff1d(all_right_valtest_indices, selected_val_right_idx, assume_unique=True)
        test_left_idx = np.setdiff1d(all_left_valtest_indices, selected_val_left_idx, assume_unique=True)
        
        split_indices = {
            'train_right_idx': selected_train_right_idx, 
            'train_left_idx': selected_train_left_idx,
            'val_right_idx': selected_val_right_idx, 
            'val_left_idx': selected_val_left_idx,
            'test_right_idx': test_right_idx, 
            'test_left_idx': test_left_idx
        }
        
        print(f"  Split indices generated: "
              f"Train (R:{len(selected_train_right_idx)}, L:{len(selected_train_left_idx)}), "
              f"Val (R:{len(selected_val_right_idx)}, L:{len(selected_val_left_idx)}), "
              f"Test (R:{len(test_right_idx)}, L:{len(test_left_idx)})")

        # Extract the data based on the indices
        train_x_right = data_right_train[:, :, selected_train_right_idx]
        train_x_left = data_left_train[:, :, selected_train_left_idx]
        
        val_x_right = data_right_valtest[:, :, selected_val_right_idx]
        val_x_left = data_left_valtest[:, :, selected_val_left_idx]
        
        time_points_dim = x_data.shape[0]
        channels_dim = x_data.shape[1]
        windowed_time_points_count = time_window_end_exclusive - time_window_start

        test_x_right = data_right_valtest[:, :, test_right_idx] if len(test_right_idx) > 0 else np.empty((time_points_dim, channels_dim, 0))
        test_x_left = data_left_valtest[:, :, test_left_idx] if len(test_left_idx) > 0 else np.empty((time_points_dim, channels_dim, 0))

        print(f"  Applying time window: {time_window_start} to {time_window_end_exclusive-1}")
        train_x_right_win = train_x_right[time_window_start:time_window_end_exclusive, :, :]
        train_x_left_win  = train_x_left[time_window_start:time_window_end_exclusive, :, :]
        val_x_right_win   = val_x_right[time_window_start:time_window_end_exclusive, :, :]
        val_x_left_win    = val_x_left[time_window_start:time_window_end_exclusive, :, :]
        
        test_x_right_win = test_x_right[time_window_start:time_window_end_exclusive, :, :] if test_x_right.shape[2] > 0 else np.empty((windowed_time_points_count, channels_dim, 0))
        test_x_left_win = test_x_left[time_window_start:time_window_end_exclusive, :, :] if test_x_left.shape[2] > 0 else np.empty((windowed_time_points_count, channels_dim, 0))

        x_train_cat = np.concatenate([train_x_right_win, train_x_left_win], axis=2)
        x_val_cat   = np.concatenate([val_x_right_win, val_x_left_win], axis=2)
        x_test_cat  = np.concatenate([test_x_right_win, test_x_left_win], axis=2)

        y_train = np.concatenate([np.full(train_x_right_win.shape[2], label_right), np.full(train_x_left_win.shape[2], label_left)])
        y_val   = np.concatenate([np.full(val_x_right_win.shape[2], label_right), np.full(val_x_left_win.shape[2], label_left)])
        y_test  = np.concatenate([np.full(test_x_right_win.shape[2], label_right), np.full(test_x_left_win.shape[2], label_left)])
        
        print(f"  Concatenated data shapes before filtering (Time,Chans,Trials): Train X={x_train_cat.shape}, Y={y_train.shape}")

        print(f"  Applying elliptical filter (lowcut={lowcut}, highcut={highcut}, fs={fs})...")
        filtered_datasets_trials_first = []
        for data_cat in [x_train_cat, x_val_cat, x_test_cat]:
            if data_cat.shape[2] == 0: # No trials in this set
                filtered_datasets_trials_first.append(np.empty((0, windowed_time_points_count, channels_dim)))
                continue
            # Transpose to (trials, time_points_windowed, channels) for iterating trials
            data_trials_first = np.transpose(data_cat, (2, 0, 1))
            processed_trials_list = [elliptical_filter(trial_data, lowcut, highcut, fs) for trial_data in data_trials_first]
            filtered_datasets_trials_first.append(np.array(processed_trials_list))

        x_train_filtered_tf, x_val_filtered_tf, x_test_filtered_tf = filtered_datasets_trials_first
        print(f"  Filtered data shapes (Trials,Time,Chans): Train={x_train_filtered_tf.shape}")

        # Transpose to Final X Shape for WGAN/CSP: (trials, channels, time_points_windowed)
        x_train_processed = np.transpose(x_train_filtered_tf, (0, 2, 1)) if x_train_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
        x_val_processed   = np.transpose(x_val_filtered_tf, (0, 2, 1)) if x_val_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
        x_test_processed  = np.transpose(x_test_filtered_tf, (0, 2, 1)) if x_test_filtered_tf.shape[0] > 0 else np.empty((0, channels_dim, windowed_time_points_count))
        print(f"  Transposed data shapes for normalization (Trials,Chans,Time): Train={x_train_processed.shape}")

        print("  Normalizing data to [-1, 1]...")
        x_train_final = _normalize_trials_internal(x_train_processed)
        x_val_final   = _normalize_trials_internal(x_val_processed)
        x_test_final  = _normalize_trials_internal(x_test_processed)
        print(f"  Final data shapes (x_..._final - Trials,Chans,Time): Train X={x_train_final.shape}")

        # Create finaltrn, finalval, finaltest dictionaries
        finaltrn = {
            'x': np.transpose(x_train_final, (2, 1, 0)) if x_train_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
            'y': y_train
        }
        finalval = {
            'x': np.transpose(x_val_final, (2, 1, 0)) if x_val_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
            'y': y_val
        }
        finaltest = {
            'x': np.transpose(x_test_final, (2, 1, 0)) if x_test_final.shape[0] > 0 else np.empty((windowed_time_points_count, channels_dim, 0)),
            'y': y_test,
            'lowfreq': lowcut,
            'highfreq': highcut
        }
        print(f"  finaltrn shapes: x={finaltrn['x'].shape}, y={finaltrn['y'].shape}")
        print(f"--- Preprocessing for Subject S{subject_id} COMPLETED ---")

        return x_train_final, y_train, finaltrn, finalval, finaltest

    except Exception as e:
        print(f"  ERROR S{subject_id} during combined preprocessing: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None, None, None, None
    
    
# --- CSP-LDA CLASSIFIER (from first script) ---
class CSPLDAClassifier:
    def __init__(self, n_components=CSP_COUNT):
        self.csp = CSP(n_components=CSP_COUNT, reg=CSP_REG, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        self.fitted = False

    def fit(self, X, y):
        # Convert labels to 1/2 format if they're in 0/1 format
        if np.min(y) == 0:
            y = y + 1

        # Convert to float64 for numerical stability
        X = X.astype(np.float64)

        # Fit CSP
        self.csp.fit(X, y.ravel())

        # Transform data
        X_csp = self.csp.transform(X)

        # Fit LDA
        self.lda.fit(X_csp, y.ravel())

        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")

        # Convert to float64
        X = X.astype(np.float64)

        # Transform with CSP
        X_csp = self.csp.transform(X)

        # Predict with LDA
        return self.lda.predict(X_csp)

    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")

        # Convert to float64
        X = X.astype(np.float64)

        # Transform with CSP
        X_csp = self.csp.transform(X)

        # Get probability estimates
        return self.lda.predict_proba(X_csp)

# Function to evaluate classification performance inside TF graph
def classification_reward(fake_data, target_class, csp_lda_classifier):
    # This function handles evaluation of CSP+LDA model inside TF graph
    def _evaluate_classification(data, target):
        # Convert tensor to numpy
        data_np = data.numpy()
        target_np = target.numpy()[0]  # Get scalar value

        # Get class probabilities from CSP+LDA
        pred_probs = csp_lda_classifier.predict_proba(data_np)

        # For target class 1 (left hand), we want low probability for class 2
        # For target class 2 (right hand), we want high probability for class 2
        if target_np == 1:
            # For left hand, we want high prob of class 1 (low prob of class 2)
            class_prob = 1.0 - pred_probs[:, 1]
        else:
            # For right hand, we want high prob of class 2
            class_prob = pred_probs[:, 1]

        # Calculate mean classification performance (higher = better)
        classification_score = np.mean(class_prob)

        # Convert to a loss (lower = better)
        cls_loss = 1.0 - classification_score

        # Debug info - print occasionally to avoid flooding output
        if np.random.random() < 0.01:  # 1% chance to print
            print(f"Target class: {target_np}, Mean probability: {classification_score:.4f}, Loss: {cls_loss:.4f}")

        return np.array(cls_loss, dtype=np.float32)

    # Use tf.py_function to wrap our NumPy-based function
    return tf.py_function(
        _evaluate_classification,
        [fake_data, tf.constant([target_class])],
        Tout=tf.float32
    )

# --- CSP-SVM Functions (from second script) ---
def csp(data):
    """
    Python implementation of Common Spatial Patterns algorithm (simplified version of CSP)
    This mimics the MATLAB csp function referenced in the code.

    Args:
        data: A structure with 'x' (EEG data) and 'y' (labels)

    Returns:
        W: CSP spatial filters
    """
    # Extract class data
    class1_indices = np.where(data['y'] == 1)[0]
    class2_indices = np.where(data['y'] == 2)[0]

    # Get data for each class
    X1 = data['x'][:, :, class1_indices]
    X2 = data['x'][:, :, class2_indices]

    # Calculate covariance matrices
    n_channels = X1.shape[1]

    # Compute covariance matrix for class 1
    cov1 = np.zeros((n_channels, n_channels))
    for trial in range(X1.shape[2]):
        E = X1[:, :, trial]
        cov1 += np.dot(E.T, E) / np.trace(np.dot(E.T, E))
    cov1 /= X1.shape[2]

    # Compute covariance matrix for class 2
    cov2 = np.zeros((n_channels, n_channels))
    for trial in range(X2.shape[2]):
        E = X2[:, :, trial]
        cov2 += np.dot(E.T, E) / np.trace(np.dot(E.T, E))
    cov2 /= X2.shape[2]

    # Solve generalized eigenvalue problem
    evals, evecs = eigh(cov1, cov1 + cov2)

    # Sort eigenvectors in descending order of eigenvalues
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Return spatial filters
    W = evecs.T

    return W

def featcrossval(finaldataset, ChRanking, numchannel):
    """
    Python implementation of the featcrossval MATLAB function

    Args:
        finaldataset: Structure with 'x' and 'y' fields
        ChRanking: Channel ranking
        numchannel: Number of channels to use

    Returns:
        producedfeatur: Extracted features
        selectedw: Selected spatial filters
    """
    # Select channels based on ranking
    a = {'x': np.zeros((finaldataset['x'].shape[0], numchannel, finaldataset['x'].shape[2])),
         'y': finaldataset['y']}

    for i in range(numchannel):
        a['x'][:, i, :] = finaldataset['x'][:, ChRanking[i], :]

    # Apply CSP
    W = csp(a)

    # Select filters based on number of channels
    if numchannel > 6:
        selectedw = np.vstack((W[0, :], W[1, :], W[2, :], W[-3, :], W[-2, :], W[-1, :]))
    else:
        selectedw = np.vstack((W[0, :], W[-1, :]))

    # Extract features
    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]

    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}

    for trial in range(ntrial):
        # Project data onto CSP filters
        selectedZ = np.dot(selectedw, a['x'][:, :, trial].T)

        # Calculate log-variance features
        variances = np.var(selectedZ, axis=1)
        producedfeatur['x'][:, trial] = np.log(variances / np.sum(variances))

    return producedfeatur, selectedw

def featcrostest(finaldataset, ChRanking, numchannel, selectedw):
    """
    Python implementation of the featcrostest MATLAB function

    Args:
        finaldataset: Structure with 'x' and 'y' fields
        ChRanking: Channel ranking
        numchannel: Number of channels to use
        selectedw: Selected spatial filters from training

    Returns:
        producedfeatur: Extracted features
    """
    # Select channels based on ranking
    a = {'x': np.zeros((finaldataset['x'].shape[0], numchannel, finaldataset['x'].shape[2])),
         'y': finaldataset['y']}

    for i in range(numchannel):
        a['x'][:, i, :] = finaldataset['x'][:, ChRanking[i], :]

    # Extract features
    ntrial = finaldataset['x'].shape[2]
    num_features = selectedw.shape[0]

    producedfeatur = {'x': np.zeros((num_features, ntrial)), 'y': finaldataset['y']}

    for trial in range(ntrial):
        # Project data onto CSP filters
        selectedZ = np.dot(selectedw, a['x'][:, :, trial].T)

        # Calculate log-variance features
        variances = np.var(selectedZ, axis=1)
        producedfeatur['x'][:, trial] = np.log(variances / np.sum(variances))

    return producedfeatur

def fitcsvm(X, Y, **kwargs):
    """
    Python implementation mimicking MATLAB's fitcsvm function

    Args:
        X: Features
        Y: Labels
        **kwargs: Additional parameters

    Returns:
        model: Trained SVM model
    """
    # Extract parameters
    standardize = kwargs.get('Standardize', False)
    kernel = kwargs.get('KernelFunction', 'linear')

    # Map MATLAB kernel names to scikit-learn kernels
    kernel_map = {
        'linear': 'linear',
        'rbf': 'rbf',
        'gaussian': 'rbf',
        'polynomial': 'poly'
    }

    # Create SVM model
    model = SVC(kernel=kernel_map.get(kernel, kernel), probability=True)

    # Standardize data if required
    if standardize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Prevent division by zero
        X = (X - X_mean) / X_std

        # Store normalization parameters in the model
        model.X_mean = X_mean
        model.X_std = X_std

    # Fit the model
    model.fit(X, Y)

    return model

def predict(model, X):
    """
    Predict function to mimic MATLAB's predict function

    Args:
        model: Trained SVM model
        X: Test features

    Returns:
        y_pred: Predicted labels
    """
    # If the model was trained with standardization, apply the same to test data
    if hasattr(model, 'X_mean') and hasattr(model, 'X_std'):
        X = (X - model.X_mean) / model.X_std

    # Make predictions
    y_pred = model.predict(X)

    return y_pred

def train_cspsvm(data):
    """Trains CSP-SVM on the given data dictionary."""
    # Extract data from dictionary format
    X_data = data['x']  # (samples, channels, trials)
    y_data = data['y']  # Labels 1 and 2
    
    n_channels = X_data.shape[1]
    
    # Channel ranking (use all channels)
    channel_ranking = np.arange(n_channels)
    
    # Apply CSP and extract features
    features, spatial_filters = featcrossval(data, channel_ranking, n_channels)
    
    # Train SVM
    X_features = features['x'].T  # (trials, features)
    y_labels = features['y']
    
    model = fitcsvm(X_features, y_labels, Standardize=True, KernelFunction='linear')
    
    return model, spatial_filters

def evaluate_cspsvm(model, spatial_filters, data):
    """Evaluates CSP-SVM on the given data dictionary."""
    # Extract data from dictionary format
    X_data = data['x']  # (samples, channels, trials)
    y_data = data['y']  # Labels 1 and 2
    
    n_channels = X_data.shape[1]
    
    # Channel ranking (use all channels)
    channel_ranking = np.arange(n_channels)
    
    # Extract features using the spatial filters from training
    features = featcrostest(data, channel_ranking, n_channels, spatial_filters)
    
    # Get predictions
    X_features = features['x'].T  # (trials, features)
    y_true = features['y']
    y_pred = predict(model, X_features)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    
    return accuracy, cm, y_pred

def evaluate_synth_on_validation(
    synth_data_batch_processed, 
    synth_labels_batch,
    x_val_real_processed,
    y_val_real,
    log_desc,
    subject_id, main_iter, gan_run_idx, batch_idx
):
    """
    Evaluates preprocessed synthetic data batch using CSP-SVM.
    Trains ONCE on synthetic data and evaluates ONCE on real validation data.
    Used ONLY for selecting the best synthetic batch.
    """
    log_prefix = f"R{main_iter}-S{subject_id} [SynthVal GAN{gan_run_idx+1}-B{batch_idx+1}]:"
    print(f"{log_prefix} Starting single evaluation of synthetic batch -> {log_desc}")

    # Validate inputs
    if synth_data_batch_processed is None or x_val_real_processed is None:
         print(f"{log_prefix} XXX ERROR: Input data is None. Skipping evaluation. XXX")
         return np.nan

    try:
        # Create data dictionary for synthetic train data
        synth_data_dict = {
            'x': synth_data_batch_processed,  # Already in the right format
            'y': synth_labels_batch           # Should already be 1, 2 format
        }
        
        # Create data dictionary for real validation data
        val_data_dict = {
            'x': x_val_real_processed,  # Already in the right format
            'y': y_val_real             # Should already be 1, 2 format
        }

        # 1. Train CSP-SVM model ONCE on the preprocessed synthetic batch
        model, spatial_filters = train_cspsvm(synth_data_dict)

        # 2. Evaluate the model ONCE on the preprocessed real validation data
        val_accuracy, _, _ = evaluate_cspsvm(model, spatial_filters, val_data_dict)
        val_accuracy_scaled = val_accuracy / 100.0 # Convert accuracy to 0-1 scale

        print(f"{log_prefix} Finished evaluation. Real Val Accuracy: {val_accuracy:.2f}%")
        return val_accuracy_scaled # Return 0-1 scale

    except Exception as e:
        print(f"{log_prefix} XXX ERROR in evaluate_synth_on_validation: {type(e).__name__} - {e} XXX")
        traceback.print_exc()
        return np.nan # Return NaN on any error

# --- WGAN-GP Components (from first script) ---
def build_generator():
    noise_input = layers.Input(shape=(100,))  # Noise vector (latent space)

    # Fully connected layer to project noise into a larger space
    x = layers.Dense(64 * 63)(noise_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((63, 64))(x)  # Shape: (63, 64)

    # Upsampling layers with Conv1DTranspose
    x = layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (126, 128)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (252, 64)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(22, kernel_size=7, strides=2, padding="same", activation="tanh")(x)  # Shape: (504, 22)

    # Crop the extra time points to match (22, 500)
    x = layers.Cropping1D(cropping=(2, 2))(x)  # Shape: (500, 22)

    # Permute to match critic's expected input shape
    x = layers.Permute((2, 1))(x)  # Shape: (22, 500)

    return tf.keras.Model(noise_input, x)

# Improved Critic
def build_critic():
    data_input = layers.Input(shape=(22, 500))
    x = layers.Permute((2, 1))(data_input)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(256, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(512, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(data_input, x)

# Gradient penalty 
def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10):
    alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        interpolated_predictions = critic(interpolated, training=True)

    gradients = gp_tape.gradient(interpolated_predictions, [interpolated])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return lambda_gp * gradient_penalty

# Main WGAN-GP training function
def train_wgan_gp_with_classification(data, target_class, epochs, batch_size, model_name, 
                                      csp_lda_classifier, lambda_cls=1.0, output_dir=BASE_OUTPUT_DIR):
    generator = build_generator()
    critic = build_critic()

    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    d_losses, g_losses, adv_losses, cls_losses = [], [], [], []
    # Training step function with classification loss
    @tf.function
    def train_step_with_classification(real_data, generator, critic, csp_lda_classifier, gen_opt, crit_opt,
                                    target_class, lambda_gp=10, lambda_cls=1.0):
        batch_size = tf.shape(real_data)[0]

        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = tf.random.normal([batch_size, 100])
            with tf.GradientTape() as tape:
                fake_data = generator(noise, training=True)
                real_output = critic(real_data, training=True)
                fake_output = critic(fake_data, training=True)
                gp_loss = gradient_penalty(critic, real_data, fake_data, lambda_gp)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp_loss

            gradients = tape.gradient(d_loss, critic.trainable_variables)
            crit_opt.apply_gradients(zip(gradients, critic.trainable_variables))

        # Train Generator
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            fake_output = critic(fake_data, training=True)

            # Adversarial loss (original objective)
            g_adv_loss = -tf.reduce_mean(fake_output)

            # Get classification loss using our wrapper
            g_cls_loss = classification_reward(fake_data, target_class, csp_lda_classifier)

            # Combined loss - use stop_gradient to prevent bad gradients
            # The loss value still affects training, but gradients won't flow through classification
            g_loss = g_adv_loss + lambda_cls * tf.stop_gradient(g_cls_loss)

        # Get gradients from adversarial loss
        gradients = tape.gradient(g_loss, generator.trainable_variables)

        # Apply gradients - scaled by classification loss
        # This scaling helps the generator improve classification more directly
        scale = 1.0 + lambda_cls * tf.stop_gradient(g_cls_loss)
        scaled_gradients = [g * scale for g in gradients]
        gen_opt.apply_gradients(zip(scaled_gradients, generator.trainable_variables))

        return d_loss, g_loss, g_adv_loss, g_cls_loss

    for epoch in range(epochs):
        # If the batch size is larger than available data, use all data
        if batch_size > data.shape[0]:
            real_data = data.astype(np.float32)
        else:
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx].astype(np.float32)

        # Execute training step
        d_loss, g_loss, adv_loss, cls_loss = train_step_with_classification(
            real_data, generator, critic, csp_lda_classifier, gen_opt, crit_opt,
            target_class, lambda_gp=LAMBDA_GP, lambda_cls=lambda_cls
        )

        # Store losses for plotting
        d_losses.append(d_loss.numpy())
        g_losses.append(g_loss.numpy())
        adv_losses.append(adv_loss.numpy())
        cls_losses.append(cls_loss.numpy())

        # Periodically print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss = {d_loss.numpy():.4f}, "
                  f"G Loss = {g_loss.numpy():.4f}, "
                  f"Adv Loss = {adv_loss.numpy():.4f}, "
                  f"Cls Loss = {cls_loss.numpy():.4f}")

    # Plot losses
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label="Critic Loss")
    plt.legend()
    plt.title(f"Critic Losses - {model_name}")

    plt.subplot(2, 2, 2)
    plt.plot(g_losses, label="Generator Total Loss")
    plt.legend()
    plt.title(f"Generator Total Losses - {model_name}")

    plt.subplot(2, 2, 3)
    plt.plot(adv_losses, label="Generator Adversarial Loss")
    plt.legend()
    plt.title(f"Generator Adversarial Losses - {model_name}")

    plt.subplot(2, 2, 4)
    plt.plot(cls_losses, label="Generator Classification Loss")
    plt.legend()
    plt.title(f"Generator Classification Losses - {model_name}")

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"WGAN_loss_{target_class}_{model_name}.png")
    try:
        plt.savefig(plot_filename)
        print(f"  Saved WGAN loss plot: {plot_filename}")
    except Exception as e:
        print(f"  Error saving WGAN loss plot: {e}")
    plt.close() # Close the figure to free memory

    return generator

# --- Synthetic Data Processing Functions ---
def generate_synthetic_data(left_generator, right_generator, num_samples_per_class, z_dim=Z_DIM):
    """Generates synthetic EEG data using trained generators."""
    if left_generator is None or right_generator is None:
        print("ERROR: Cannot generate synthetic data, generator(s) missing.")
        return None, None
    try:
        # Generate noise
        noise_l = tf.random.normal([num_samples_per_class, z_dim])
        noise_r = tf.random.normal([num_samples_per_class, z_dim])
        # Generate data (set training=False for inference mode)
        synth_l = left_generator(noise_l, training=False)
        synth_r = right_generator(noise_r, training=False)
        # Combine and create labels
        synth_data = tf.concat([synth_l, synth_r], axis=0).numpy()
        synth_labels = np.concatenate([np.ones(num_samples_per_class, dtype=int), # Class 1
                                    np.ones(num_samples_per_class, dtype=int) * 2]) # Class 2
        # Shuffle
        indices = np.arange(synth_data.shape[0])
        np.random.shuffle(indices)
        synth_data, synth_labels = synth_data[indices], synth_labels[indices]
        print(f"  Generated synthetic data: {synth_data.shape}, Labels: {synth_labels.shape}")
        # Check for and handle NaN/Inf in output
        nan_count = np.sum(np.isnan(synth_data))
        inf_count = np.sum(np.isinf(synth_data))
        if nan_count > 0 or inf_count > 0:
            print(f"WARN: Generated synthetic data contains {nan_count} NaNs and {inf_count} Infs! Applying nan_to_num.")
            synth_data = np.nan_to_num(synth_data) # Replace NaN/Inf
        return synth_data, synth_labels
    except Exception as e:
        print(f"ERROR during synthetic data generation: {e}")
        return None, None


# --- Main Execution Loop ---
def main():
    overall_results_list = []
    start_time_total = time.time()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # # --- Set Global Seed ---
    # print(f"*** Setting global random seed: {RANDOM_SEED} ***")
    # np.random.seed(RANDOM_SEED)
    # tf.random.set_seed(RANDOM_SEED)

    # --- Load Data ---
    print("Loading data...")
    try:
        data = scipy.io.loadmat('data1.mat')
        subjects_struct = data['xsubi_all'][0]
        num_subjects = len(subjects_struct)
        print(f"Found {num_subjects} subjects.")
    except FileNotFoundError:
        print("ERROR: data1.mat not found.")
        return
    except KeyError as e:
        print(f"ERROR: Could not find key {e} in the loaded .mat file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Loading data...")
    try:
        vt_data = scipy.io.loadmat('data2.mat')
        vt_subjects_struct = vt_data['txsubi_all'][0]
        vt_num_subjects = len(vt_subjects_struct)
        print(f"Found {vt_num_subjects} subjects.")
    except FileNotFoundError:
        print("ERROR: data1.mat not found.")
        return
    except KeyError as e:
        print(f"ERROR: Could not find key {e} in the loaded .mat file.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    warnings.filterwarnings('ignore')

    for main_iter in range(1, NUM_MAIN_ITERATIONS + 1):
        print(f"\n{'#'*80}\n### MAIN ITERATION {main_iter}/{NUM_MAIN_ITERATIONS} ###\n{'#'*80}")
        run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{main_iter}")
        os.makedirs(run_dir, exist_ok=True)
        start_time_run = time.time()
        run_subject_summaries = []

        for subject_idx in range(num_subjects):
            subject_id = subject_idx + 1
            log_prefix = f"R{main_iter}-S{subject_id}:"
            print(f"\n{'='*70}\n {log_prefix} PROCESSING SUBJECT {subject_id}/{num_subjects} \n{'='*70}")
            start_time_subject = time.time()
            subject_dir = os.path.join(run_dir, f"subject_{subject_id}")
            os.makedirs(subject_dir, exist_ok=True)

            # Clear TF session and garbage collection
            tf.keras.backend.clear_session()
            gc.collect()

            # Initialize summary dict
            subject_summary = {
                'main_iter': main_iter, 'subject_id': subject_id,
                'real_accuracy_final_test': np.nan, 
                'synthetic_only_test_acc': np.nan,
                'best_synth_gan_run': np.nan,
                'best_synth_batch_idx': np.nan,
                'best_mixed_test_acc': np.nan,
                'best_mix_ratio': np.nan,
                'best_synth_val_score': np.nan,
                'log': [], 'notes': ''
            }

            # --- Step 1: Preprocessing for Evaluation (CSP-SVM) ---
            print(f"\n{log_prefix} STEP 1a: Preprocessing data for CSP-SVM evaluation...")
             # --- Step 1b: Preprocessing for WGAN Training (first set of trials) ---
            print(f"\n{log_prefix} STEP 1b: Preprocessing data for WGAN training (using first {N_TRAIN_PER_CLASS} trials)...")

            try:
                # Get subject data structure
                xsubi1 = subjects_struct[subject_idx]
                xsubi2 = vt_subjects_struct[subject_idx]
                # Get raw subject data for WGAN preprocessing
                subject_data = {
                    'x': np.concatenate([xsubi1['Right'], xsubi1['Left']], axis=2),
                    'y': np.concatenate([np.ones(xsubi1['Right'].shape[2])*2, np.ones(xsubi1['Left'].shape[2])])
                }
                vt_subject_data = {
                    'x': xsubi2['x'],
                    'y': xsubi2['y']
                }
                
                # Split data using random selection approach for CSP-SVM evaluation
                x_train_wgan, y_train_wgan, finaltrn, finalval, finaltest = eeg_preprocess(
                    vt_subject_data['x'], vt_subject_data['y'], subject_data['x'], subject_data['y'], subject_id=subject_id, train_trials_per_class=None,val_trials_per_class=N_VAL_PER_CLASS, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, time_window_start=115, time_window_end_exclusive=615, label_left=1, label_right=2
                )
                
                if finaltrn is None: 
                    raise ValueError("Preprocessing for evaluation returned None.")
                
                # These are now the data dictionaries
                X_train = finaltrn
                X_val = finalval
                X_test = finaltest
                
                print(f"{log_prefix} Preprocessing & Split for evaluation successful. Train={X_train['x'].shape}, Val={X_val['x'].shape}, Test={X_test['x'].shape}")
                print(f"{log_prefix} Preprocessing for WGAN training successful. WGAN Train data shape: {x_train_wgan.shape}")

            except Exception as e:
                print(f"{log_prefix} ERROR in Step 1a (Preprocessing for eval): {type(e).__name__} - {e}")
                traceback.print_exc()
                subject_summary['notes'] = f'Preproc err: {e}'
                subject_summary['log'].append(f"ERROR Step 1a: {e}")
                run_subject_summaries.append(subject_summary)
                print(f"{log_prefix} ERROR in Step 1b (Preprocessing for WGAN): {type(e).__name__} - {e}")
                traceback.print_exc()
                subject_summary['notes'] += f'WGANPreproc err: {e};'
                subject_summary['log'].append(f"ERROR Step 1b: {e}")
                run_subject_summaries.append(subject_summary)
                continue # Skip subject

            # --- Step 2: Evaluate Real Data Only (Single Run on Test Set) ---
            print(f"\n{log_prefix} STEP 2: Evaluating REAL Data Only (Train->Test)...")
            try:
                # Train ONCE on X_train (preprocessed) - already in correct format
                model_real, filters_real = train_cspsvm(X_train)

                # Evaluate ONCE on X_test (preprocessed)
                real_test_acc, _, _ = evaluate_cspsvm(model_real, filters_real, X_test)
                real_test_acc_scaled = real_test_acc / 100.0 # Scale 0-1

                subject_summary['real_accuracy_final_test'] = real_test_acc_scaled
                print(f"{log_prefix} ---> REAL Data FINAL Test Accuracy: {real_test_acc:.2f}%")

                if np.isnan(real_test_acc_scaled):
                    subject_summary['log'].append("RealOnly FinalTest Eval failed.")
                    raise ValueError("Real data evaluation failed, cannot proceed.")

                del model_real, filters_real # Clean up

            except Exception as e:
                print(f"{log_prefix} ERROR in Step 2 (Real Eval): {type(e).__name__} - {e}")
                traceback.print_exc()
                subject_summary['log'].append(f"ERROR Step 2: {e}")
                run_subject_summaries.append(subject_summary)
                continue # Skip subject if baseline fails

            # --- Step 3: Prepare Data for WGAN Training ---
            print(f"\n{log_prefix} STEP 3a: Preparing Data for WGAN Training...")
            try:
                # Split WGAN training data by class (using the data from preprocess_for_wgan)
                left_indices_wgan = np.where(y_train_wgan == 1)[0]   # Class 1 indices
                right_indices_wgan = np.where(y_train_wgan == 2)[0]  # Class 2 indices
                
                if len(left_indices_wgan) == 0 or len(right_indices_wgan) == 0:
                    print(f"{log_prefix} ERROR: Missing training data for one/both classes in WGAN training set.")
                    subject_summary['notes'] += "WGANFail_NoClassDataTrain;"
                    subject_summary['log'].append("ERROR Step 3: Missing class data in WGAN train")
                    run_subject_summaries.append(subject_summary)
                    continue
                
                # Get left and right hand data for training - these are already in (trials, channels, samples) format
                left_hand_real_train = x_train_wgan[left_indices_wgan]
                right_hand_real_train = x_train_wgan[right_indices_wgan]
                
                print(f"{log_prefix} Extracted Left hand data for WGAN: {left_hand_real_train.shape}, "
                    f"Right hand data for WGAN: {right_hand_real_train.shape}")
                
                # --- Create and train CSP-LDA classifier for WGAN guidance ---
                print(f"{log_prefix} STEP 3b: Training CSP-LDA Classifier for WGAN Guidance...")
                
                # Combine left and right hand data for CSP-LDA classifier training
                X_combined = np.concatenate([left_hand_real_train, right_hand_real_train], axis=0)
                y_combined = np.concatenate([np.ones(len(left_hand_real_train)), 
                                        np.ones(len(right_hand_real_train)) * 2])
                
                # Create and train CSP-LDA classifier with specific parameters
                csp_lda_classifier = CSPLDAClassifier(n_components=CSP_COUNT)  # Using specified parameters
                csp_lda_classifier.fit(X_combined, y_combined)
                
                print(f"{log_prefix} CSP-LDA Classifier trained successfully.")

            except Exception as e:
                print(f"{log_prefix} ERROR preparing data or training CSP-LDA for WGAN: {type(e).__name__} - {e}")
                traceback.print_exc()
                subject_summary['notes'] += f"CSPLDATrainErr:{e};"
                subject_summary['log'].append(f"ERROR Step 3: CSP-LDA train: {e}")
                run_subject_summaries.append(subject_summary)
                continue # Skip subject
        
            # --- WGAN Training Loop ---
            stored_generator_weights = []
            
            print(f"\n{log_prefix} STEP 3c: Starting {NUM_GAN_RUNS} WGAN Training Runs with CSP-LDA guidance...")
            
            for gan_run_idx in range(NUM_GAN_RUNS):
                print(f"\n{log_prefix} ===== WGAN Training Run {gan_run_idx+1}/{NUM_GAN_RUNS} =====")
                left_generator = None
                right_generator = None
                tf.keras.backend.clear_session()
                gc.collect()
                try:
                    # Train left hand generator with class 1 guidance
                    print(f"{log_prefix} Training LEFT hand generator with class 1 guidance...")
                    left_generator = train_wgan_gp_with_classification(
                        left_hand_real_train, 1, GAN_EPOCHS, BATCH_SIZE, 
                        f'Left_Run{gan_run_idx+1}', 
                        csp_lda_classifier, lambda_cls=LAMBDA_CLS, output_dir=subject_dir
                    )
                    
                    # Train right hand generator with class 2 guidance
                    print(f"{log_prefix} Training RIGHT hand generator with class 2 guidance...")
                    right_generator = train_wgan_gp_with_classification(
                        right_hand_real_train, 2, GAN_EPOCHS, BATCH_SIZE, 
                        f'Right_Run{gan_run_idx+1}', 
                        csp_lda_classifier, lambda_cls=LAMBDA_CLS, output_dir=subject_dir
                    )

                    if left_generator and right_generator:
                        # Store weights immediately after successful training
                        stored_generator_weights.append((copy.deepcopy(left_generator.get_weights()), 
                                                      copy.deepcopy(right_generator.get_weights())))
                        print(f"{log_prefix} --- Completed WGAN Run {gan_run_idx+1}. Stored weights. ---")
                    else:
                        stored_generator_weights.append((None, None)) # Store None if failed
                        print(f"{log_prefix} --- WGAN Run {gan_run_idx+1} failed (generator not returned). ---")
                        subject_summary['log'].append(f"WGANRun{gan_run_idx+1}Fail;")
                        subject_summary['notes']+=f"WGANRun{gan_run_idx+1}Fail;"

                except Exception as e:
                    print(f"{log_prefix} ERROR during WGAN Run {gan_run_idx+1}: {type(e).__name__} - {e}")
                    traceback.print_exc()
                    stored_generator_weights.append((None, None)) # Store None on error
                    subject_summary['log'].append(f"ERROR WGANRun{gan_run_idx+1}: {e};")
                    subject_summary['notes'] += f"WGANRun{gan_run_idx+1}ERR:{e};"
                finally:
                    # Ensure generators are deleted even if training failed mid-way
                    del left_generator, right_generator
                    gc.collect()
            # --- End WGAN Training Loop ---

            # --- Step 4: Synthetic Batch Generation and Selection (using Val data) ---
            print(f"\n{log_prefix} STEP 4: Generating & Selecting Best Synthetic Batch (using validation data)...")
            best_val_acc_synth_selection = -1.0 # Internal score for selection
            best_batch_data_preprocessed = None # Store PREPROCESSED best batch as a dictionary
            best_batch_gan_run_idx = -1
            best_batch_idx = -1

            temp_left_gen = None
            temp_right_gen = None
            # Determine num_synth_per_class based on train set size
            num_synth_per_class_for_eval = N_TRAIN_PER_CLASS * 2

            for gan_run_idx, (left_weights, right_weights) in enumerate(stored_generator_weights):
                if left_weights is None or right_weights is None:
                    print(f"{log_prefix} Skipping generation/validation for failed GAN Run {gan_run_idx+1}.")
                    continue

                print(f"\n{log_prefix} ===== Evaluating Batches from GAN Run {gan_run_idx+1}/{NUM_GAN_RUNS} =====")
                try: # Load weights into temporary generators
                    if temp_left_gen is None: temp_left_gen = build_generator()
                    if temp_right_gen is None: temp_right_gen = build_generator()
                    temp_left_gen.set_weights(left_weights)
                    temp_right_gen.set_weights(right_weights)
                except Exception as e:
                    print(f"{log_prefix} ERROR setting generator weights for GAN Run {gan_run_idx+1}: {e}")
                    continue

                # --- Generation and Evaluation Loop (per batch) ---
                for batch_idx in range(NUM_BATCHES_PER_RUN):
                    print(f"{log_prefix} --- GAN Run {gan_run_idx+1}, Batch {batch_idx+1}/{NUM_BATCHES_PER_RUN} ---")
                    synth_data_batch_raw, synth_labels_batch = None, None # Raw output
                    synth_data_batch_processed = None           # Processed output

                    # 1. Generate Raw Data
                    try:
                        synth_data_batch_raw, synth_labels_batch = generate_synthetic_data(
                            temp_left_gen, temp_right_gen, num_synth_per_class_for_eval, z_dim=Z_DIM)
                        if synth_data_batch_raw is None: 
                            raise ValueError("Generation returned None.")
                    except Exception as e:
                        print(f"{log_prefix} ERROR generating batch: {e}")
                        subject_summary['log'].append(f"ERROR GenB{gan_run_idx+1}-{batch_idx+1}: {e}")
                        continue

                    # 2. Preprocess Synthetic Batch using train_mean, train_std
                    print(f"{log_prefix} Preprocessing synthetic batch G{gan_run_idx+1}-B{batch_idx+1}...")
                    try:
                        
                        # Transpose processed data to match the format needed for CSP-SVM
                        synth_data_transposed = np.transpose(synth_data_batch_raw, (2, 1, 0))  # (samples, channels, trials)
                        
                        print(f"{log_prefix} Processed synthetic data shape: {synth_data_batch_raw.shape} -> transposed: {synth_data_transposed.shape}")
                        
                    except Exception as e:
                        print(f"{log_prefix} ERROR Preprocessing failed for batch: {e}")
                        subject_summary['log'].append(f"ERROR PreprocSynthB{gan_run_idx+1}-{batch_idx+1}: {e}")
                        continue

                    # 3. Evaluate using CSP-SVM on validation data
                    try:
                        # Pass the transposed data to match the expected format
                        current_batch_val_acc = evaluate_synth_on_validation(
                            synth_data_transposed,     # Transposed to (samples, channels, trials)
                            synth_labels_batch,        # Labels array
                            X_val['x'],                # Validation data (already in correct format)
                            X_val['y'],                # Validation labels
                            f"GAN{gan_run_idx+1}_B{batch_idx+1}",
                            subject_id, main_iter, gan_run_idx, batch_idx
                        )

                        # Update best batch logic (store both original and transposed data)
                        if not np.isnan(current_batch_val_acc) and current_batch_val_acc > best_val_acc_synth_selection:
                            print(f"{log_prefix} *** New Best Synthetic Batch Found (Internal Val Acc {current_batch_val_acc*100:.2f}%)! ***")
                            best_val_acc_synth_selection = current_batch_val_acc
                            
                            # Store both formats for flexibility
                            best_batch_data_preprocessed = {
                                'original': synth_data_batch_raw.copy(),
                                'transposed': synth_data_transposed.copy(),
                                'labels': synth_labels_batch.copy()
                            }
                            best_batch_gan_run_idx = gan_run_idx
                            best_batch_idx = batch_idx
                        elif np.isnan(current_batch_val_acc):
                            subject_summary['log'].append(f"ValB{gan_run_idx+1}-{batch_idx+1} NaN") 

                    except Exception as e:
                        print(f"{log_prefix} ERROR during validation of batch G{gan_run_idx+1}-B{batch_idx+1}: {e}")
                        subject_summary['log'].append(f"ERROR ValB{gan_run_idx+1}-{batch_idx+1}: {e}")

                # --- End Batch Loop ---
            # --- End GAN Run Loop ---

            # Clean up temp generators
            del temp_left_gen, temp_right_gen
            gc.collect()

            # Store the validation score that led to the best batch selection
            subject_summary['best_synth_val_score'] = best_val_acc_synth_selection
            subject_summary['best_synth_gan_run'] = best_batch_gan_run_idx + 1 if best_batch_gan_run_idx != -1 else np.nan
            subject_summary['best_synth_batch_idx'] = best_batch_idx + 1 if best_batch_idx != -1 else np.nan

            # STEP 5: Final evaluations with best synthetic batch
            if best_batch_data_preprocessed is not None and best_val_acc_synth_selection > 0:
                print(f"\n{log_prefix}: STEP 5: Using BEST PREPROCESSED Synthetic Batch (GAN Run {best_batch_gan_run_idx+1}, Batch {best_batch_idx+1}) for Final Evals.")
                print(f"{log_prefix}: (Selected based on validation acc: {best_val_acc_synth_selection*100:.2f}%)")
                
                # Extract data from the dictionary format
                X_synth_best_proc = best_batch_data_preprocessed['transposed']  # Get the transposed version for CSP-SVM
                y_synth_best = best_batch_data_preprocessed['labels']

                # --- 5a: Final Evaluation - Synth Only (Single Run on Test Data) ---
                print(f"\n{log_prefix} ---> STEP 5a: Evaluating Best Synthetic Batch (Synth Only) on FINAL TEST Set...")
                try:
                    # Create synthetic data dict - synth data is already preprocessed
                    synth_data_dict = {
                        'x': X_synth_best_proc,
                        'y': y_synth_best
                    }
                    
                    # Train ONCE on synthetic data
                    model_synth, filters_synth = train_cspsvm(synth_data_dict)

                    # Evaluate ONCE on test data
                    synth_only_test_acc, _, _ = evaluate_cspsvm(model_synth, filters_synth, X_test)
                    synth_only_test_acc_scaled = synth_only_test_acc / 100.0

                    subject_summary['synthetic_only_test_acc'] = synth_only_test_acc_scaled
                    if np.isnan(synth_only_test_acc_scaled):
                        subject_summary['log'].append("FinalSynthOnlyEval NaN")
                        subject_summary['notes']+="FinalSynthEvalNaN;"
                        print(f"{log_prefix} ---> FINAL Synth Only TEST Accuracy: NaN")
                    else:
                        print(f"{log_prefix} ---> FINAL Synth Only TEST Accuracy: {synth_only_test_acc:.2f}%")

                    del model_synth, filters_synth # Clean up

                except Exception as e:
                    print(f"{log_prefix} ERROR Step 5a (Final Synth Eval): {e}")
                    traceback.print_exc()
                    subject_summary['log'].append(f"ERROR Step 5a: {e}")
                    subject_summary['notes']+=f"FinalSynthEvalERR:{e};"

                # --- 5b: Final Evaluation - Mixed Ratios (Single Run on Test Data) ---
                print(f"\n{log_prefix} ---> STEP 5b: Evaluating Mixed Ratios on FINAL TEST Set...")
                mixed_accs_test = {} # Store {ratio: final_test_accuracy}

                for ratio in MIX_RATIOS:
                    print(f"{log_prefix} ----- Evaluating Mix Ratio: {ratio}% -----")
                    try:
                        # Determine num synthetic samples to add based on TRAIN size
                        num_real_train = X_train['y'].shape[0]  # Use number of training labels
                        num_synth_available = y_synth_best.shape[0]
                        num_add = min(int(num_real_train * (ratio / 100.0)), num_synth_available)

                        # Use Real Only result if ratio is 0 or num_add is 0
                        if ratio == 0 or num_add <= 0:
                            print(f"{log_prefix} Mix {ratio}% (num_add={num_add}): Using Real Only Test Accuracy.")
                            mixed_accs_test[ratio] = subject_summary['real_accuracy_final_test']
                            continue

                        print(f"{log_prefix} DEBUG - X_train['x']: {X_train['x'].shape}, X_synth_best_proc: {X_synth_best_proc.shape}")
                        
                        # Create mixed dataset
                        if X_synth_best_proc.shape[2] >= num_add:
                            # Data is already in (samples, channels, trials) format
                            synth_subset = X_synth_best_proc[:, :, :num_add]
                            print(f"{log_prefix} Using existing transposed synthetic data: {synth_subset.shape}")
                        else:
                            # Need to transpose from (trials, channels, samples) to (samples, channels, trials)
                            synth_subset = np.transpose(best_batch_data_preprocessed['original'][:num_add], (2, 1, 0))
                            print(f"{log_prefix} Transposed synthetic data from {best_batch_data_preprocessed['original'][:num_add].shape} to {synth_subset.shape}")
                            
                        # Combine real and synthetic data
                        combined_x = np.concatenate([X_train['x'], synth_subset], axis=2)  # Concatenate along trials dimension
                        combined_y = np.concatenate([X_train['y'], y_synth_best[:num_add]])  # Combine labels
                        
                        # Create mixed data dictionary
                        X_mix = {
                            'x': combined_x,
                            'y': combined_y
                        }
                        
                        print(f"{log_prefix} Mix {ratio}%: TrainReal={num_real_train}, SynthAdded={num_add}, TotalTrain={X_mix['x'].shape[2]}")

                        # Shuffle mixed training data (need to keep x and y aligned)
                        shuffle_idx = np.random.permutation(X_mix['x'].shape[2])
                        X_mix['x'] = X_mix['x'][:, :, shuffle_idx]
                        X_mix['y'] = X_mix['y'][shuffle_idx]

                        # Train ONCE on X_mix
                        model_mix, filters_mix = train_cspsvm(X_mix)

                        # Evaluate ONCE on X_test (preprocessed)
                        mix_test_acc, _, _ = evaluate_cspsvm(model_mix, filters_mix, X_test)
                        mix_test_acc_scaled = mix_test_acc / 100.0
                        mixed_accs_test[ratio] = mix_test_acc_scaled

                        if np.isnan(mix_test_acc_scaled):
                            subject_summary['log'].append(f"FinalMix{ratio}% NaN") 
                            subject_summary['notes']+=f"FinalMix{ratio}%NaN;"
                            print(f"{log_prefix} Mix {ratio}% FINAL TEST Accuracy: NaN")
                        else:
                            print(f"{log_prefix} Mix {ratio}% FINAL TEST Accuracy: {mix_test_acc:.2f}%")

                        # Clean up
                        del model_mix, filters_mix, X_mix, combined_x, combined_y, synth_subset

                    except Exception as e:
                        print(f"{log_prefix} ERROR Step 5b (Mix {ratio}% Eval): {e}")
                        traceback.print_exc()
                        subject_summary['log'].append(f"ERROR Step 5b Mix{ratio}: {e}")
                        subject_summary['notes']+=f"FinalMix{ratio}%ERR:{e};"
                        mixed_accs_test[ratio] = np.nan # Mark failure
                # --- End Mix Ratio Loop ---

                # --- Find Best Mix Ratio based on FINAL TEST accuracy ---
                valid_mixed_accs_test = {r: a for r, a in mixed_accs_test.items() if not np.isnan(a)}
                if valid_mixed_accs_test:
                    # Find the ratio with the maximum accuracy
                    best_mix_ratio = max(valid_mixed_accs_test, key=valid_mixed_accs_test.get)
                    subject_summary['best_mixed_test_acc'] = valid_mixed_accs_test[best_mix_ratio]
                    subject_summary['best_mix_ratio'] = best_mix_ratio
                    print(f"\n{log_prefix} Best Mix Ratio = {best_mix_ratio}% (Final Test Acc: {valid_mixed_accs_test[best_mix_ratio]*100:.2f}%)")
                else:
                    print(f"\n{log_prefix} No mixed ratio evaluations completed successfully.")
                    subject_summary['notes'] += 'NoMixResults;'

                del X_synth_best_proc, y_synth_best # Clean up best batch data

            else: # No best_batch_data_preprocessed found in Step 4
                print(f"\n{log_prefix} SKIPPING Steps 5a & 5b: No best synthetic batch was selected or preprocessed successfully.")
                subject_summary['log'].append("SKIPPED Steps 5a, 5b - No best synth batch.")
                subject_summary['notes'] += 'NoBestSynthBatch;'

            # --- Step 6: Save Best WGAN Generators ---
            print(f"\n{log_prefix} STEP 6: Saving Best WGAN Generators...")
            if subject_summary['best_synth_gan_run'] is not np.nan and not np.isnan(subject_summary['best_synth_gan_run']):
                best_gan_run_idx_save = int(subject_summary['best_synth_gan_run'] - 1) # Convert back to 0-based index
                if 0 <= best_gan_run_idx_save < len(stored_generator_weights):
                    best_weights = stored_generator_weights[best_gan_run_idx_save]
                    if best_weights[0] is not None and best_weights[1] is not None:
                        try:
                            best_left_gen = build_generator()
                            best_right_gen = build_generator()
                            best_left_gen.set_weights(best_weights[0])
                            best_right_gen.set_weights(best_weights[1])
                            
                            save_name_l = os.path.join(subject_dir, f"best_left_generator_run{best_gan_run_idx_save+1}.h5")
                            save_name_r = os.path.join(subject_dir, f"best_right_generator_run{best_gan_run_idx_save+1}.h5")
                            best_left_gen.save(save_name_l)
                            best_right_gen.save(save_name_r)
                            print(f"{log_prefix} Successfully saved best generator models (Run {best_gan_run_idx_save+1}).")
                            del best_left_gen, best_right_gen
                        except Exception as e: 
                            print(f"{log_prefix} ERROR saving best generators: {e}")
                    else: 
                        print(f"{log_prefix} Cannot save best generators: weights are None for run {best_gan_run_idx_save+1}.")
                else: 
                    print(f"{log_prefix} Best GAN run index {best_gan_run_idx_save} out of bounds for saving.")
            else: 
                print(f"{log_prefix} No best GAN run index found to save generators.")

            # --- Subject Cleanup & Summary ---
            # Clean up main data
            del X_train, X_val, X_test

            # Clean up GAN-related variables
            if 'stored_generator_weights' in locals(): del stored_generator_weights
            if 'left_hand_real_train' in locals(): del left_hand_real_train
            if 'right_hand_real_train' in locals(): del right_hand_real_train
            if 'csp_lda_classifier' in locals(): del csp_lda_classifier

            # Clean up synthetic data variables
            if 'best_batch_data_preprocessed' in locals() and best_batch_data_preprocessed is not None:
                del best_batch_data_preprocessed
            if 'synth_data_batch_raw' in locals(): del synth_data_batch_raw
            if 'synth_labels_batch' in locals(): del synth_labels_batch

            gc.collect() # Force garbage collection

            run_subject_summaries.append(subject_summary)
            subject_elapsed = time.time() - start_time_subject
            print(f"\n--- {log_prefix} Finished Subject {subject_id} in {subject_elapsed/60:.1f} min ---")
            tf.keras.backend.clear_session()
            gc.collect() # Extra cleanup

        # --- End Subject Loop ---
        overall_results_list.extend(run_subject_summaries)
        run_elapsed = time.time() - start_time_run
        print(f"\n### MAIN ITERATION {main_iter} Finished in {run_elapsed / 60:.2f} min ###")

    # --- End Main Iteration Loop ---

    # --- Final Aggregation and Plotting ---
    print(f"\n{'='*80}\n FINAL SUMMARY \n{'='*80}")
    total_elapsed = time.time() - start_time_total
    print(f" Total script time: {total_elapsed / 60:.2f} min ({total_elapsed / 3600:.2f} hours)")

    if overall_results_list:
        results_df = pd.DataFrame(overall_results_list)
        if 'log' in results_df.columns:
            results_df['log'] = results_df['log'].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) and x else '')
        results_df = results_df.set_index(['main_iter', 'subject_id']).sort_index()

        csv_filename = os.path.join(BASE_OUTPUT_DIR, "ALL_RUNS_Combined_Summary_Final.csv")
        try: 
            results_df.to_csv(csv_filename)
            print(f"Saved final combined summary: {csv_filename}")
        except Exception as e: 
            print(f"ERR saving final CSV: {e}")

        # Generate summary plots
        cols_to_show = ['real_accuracy_final_test', 'synthetic_only_test_acc',
                        'best_mixed_test_acc', 'best_mix_ratio',
                        'best_synth_val_score', 'best_synth_gan_run', 'best_synth_batch_idx']
        avg_cols = ['real_accuracy_final_test', 'synthetic_only_test_acc', 'best_mixed_test_acc']
        
        try:
            print("\nCombined Summary Table Sample:")
            print(results_df[cols_to_show].round(4).head(20))
            
            avg_results = results_df.groupby('subject_id')[avg_cols].mean().dropna(how='all')
            if not avg_results.empty:
                print("\nAverage TEST Accuracy Per Subject Across Runs:")
                print(avg_results.round(4))
                
                # Create bar plot
                plot_final_comparison(avg_results, BASE_OUTPUT_DIR, NUM_MAIN_ITERATIONS)
        except Exception as e:
            print(f"Error in summary statistics: {e}")
    else:
        print("No results generated.")

    print("\n--- Script Finished ---")

def plot_final_comparison(avg_results, output_dir, num_iterations):
    """Generate bar plot comparing real, synthetic, and mixed results."""
    num_subj = len(avg_results)
    subj_ids = avg_results.index.astype(str)
    x_pos = np.arange(num_subj)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(10, num_subj * 0.7), 6))
    
    # Plot bars for each metric
    metrics = [
        ('real_accuracy_final_test', 'Real Only (Final Test)', 'cornflowerblue'),
        ('synthetic_only_test_acc', 'Best Synth (Final Test)', 'lightgreen'),
        ('best_mixed_test_acc', 'Best Mixed (Final Test)', 'salmon')
    ]
    
    for i, (col, label, color) in enumerate(metrics):
        if col in avg_results.columns:
            ax.bar(x_pos + (i-1)*width, avg_results[col].fillna(0), width, label=label, color=color)
    
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Avg Test Accuracy (CSP-SVM)')
    ax.set_title(f'Average CSP-SVM Test Accuracy Comparison ({num_iterations} Run(s))')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subj_ids, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize='small', loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    fig.tight_layout()
    plot_filename = os.path.join(output_dir, "ALL_RUNS_Avg_Test_Accuracy_Comparison_Final.png")
    try:
        plt.savefig(plot_filename)
        print(f"\nSaved final comparison plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving final plot: {e}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()
