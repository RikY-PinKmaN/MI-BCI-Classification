# analysis_with_fid.py
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import scipy.io
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import ellip, filtfilt, ellipord, spectrogram, welch # Added spectrogram, welch
from scipy.fft import fft
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import sqrtm # For FID calculation
# Use tf.image for resizing if available and preferred
# from skimage.transform import resize # Alternative for resizing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pandas as pd
import traceback

# --- Configuration ---
# #################### SPECIFY THESE ####################
SUBJECT_ID_TO_ANALYZE = 9   # <<< CHANGE THIS to the subject you want to analyze
BASE_OUTPUT_DIR = "WGAN_GP_CL_S1_20tr_per_class" # <<< Must match the output dir of training
# (Adjust this path if your script is not in the same directory as the output folder)
# ######################################################

# Constants copied/derived from training script (ensure these match your training setup)
FS = 250  # Sampling Rate
LOWCUT = 8
HIGHCUT = 35
Z_DIM = 100
# Indices for C3 and C4 (adjust if your channel configuration is different)
C3_IDX = 7 # Example: Often associated with right-hand MI
C4_IDX = 11 # Example: Often associated with left-hand MI
CHANNELS_TO_ANALYZE = {'C3 (Right Hand)': C3_IDX, 'C4 (Left Hand)': C4_IDX}

# Analysis-specific parameters
N_SYNTH_SAMPLES_PER_CLASS = 70 # Number of synthetic samples to generate per class for analysis
CONFIDENCE_LEVEL = 0.95
TF_NPERSEG = 64 # Window size for STFT
TF_NOVERLAP = 48 # Overlap for STFT (75% of nperseg)
TF_FREQ_RANGE = (1, 50) # Freq range for TF plots
PSD_NPERSEG = 256 # Window size for Welch PSD
PSD_NOVERLAP = 128 # Overlap for Welch PSD
PSD_FREQ_RANGE = (1, 50)  # Freq range for PSD plots
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50) # Make sure upper limit aligns with PSD_FREQ_RANGE if needed
}

# --- FID Specific Configuration ---
FID_IMAGE_SIZE = (75, 75)  # Target size for InceptionV3 input (must be >= 75x75)
FID_NPERSEG = 64           # Spectrogram window size for FID images
FID_NOVERLAP = 48          # Spectrogram overlap for FID images
FID_EPS = 1e-6             # Epsilon for numerical stability in FID calculation

# --- Output Setup ---
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"Analysis_subject_{SUBJECT_ID_TO_ANALYZE}")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
print(f"Analysis results will be saved in: {ANALYSIS_OUTPUT_DIR}")

# --- Helper Functions (Copied/Adapted from Training and Snippets) ---

def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    """Applies an elliptical bandpass filter."""
    nyq = 0.5 * fs
    wp = [lowcut / nyq, highcut / nyq]
    ws = [(lowcut - 1) / nyq, (highcut + 1) / nyq]
    try:
        n, wn = ellipord(wp, ws, rp, rs)
        b, a = ellip(n, rp, rs, wn, btype='band')
        filtered_data = filtfilt(b, a, data, axis=0)
    except ValueError as e:
        print(f"Filter warning: {e}. Trying lower order filter.")
        # Fallback to a simpler filter if ellipord fails (e.g., for very short signals)
        order=4 # Example fixed order
        try:
            b, a = ellip(order, rp, rs, wp, btype='band')
            filtered_data = filtfilt(b, a, data, axis=0)
        except Exception as fallback_e:
             print(f"Fallback filter also failed: {fallback_e}. Returning original data.")
             filtered_data = data # Return unfiltered data as last resort
    return filtered_data

def _normalize_trials_internal(data):
    """Normalizes each trial in the dataset individually to [-1, 1]."""
    normalized_data = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]): # Iterate through trials
        trial = data[i]
        min_val, max_val = np.min(trial), np.max(trial)
        if max_val - min_val > 1e-6:
             normalized_data[i] = 2 * (trial - min_val) / (max_val - min_val) - 1
        else:
             normalized_data[i] = trial - np.mean(trial) # Center around 0 if range is too small
    return normalized_data

def preprocess_training_data(x_data, y_data, subject_id):
    """
    Preprocesses the *entire* input dataset for TRAINING with enhanced error reporting.
    (Assuming N_CHANNELS and N_TIMEPOINTS are determined during preprocessing)
    """
    print(f"S{subject_id}: Preprocessing ALL data for Training...")
    try:
        print(f"S{subject_id}: Raw data shape: x={x_data.shape}, y={y_data.shape}")
    except:
        print(f"S{subject_id}: ERROR - Could not get raw data shape.")
        return None, None, None, None # Return None for N_CHANNELS, N_TIMEPOINTS

    # Process labels
    try:
        y_data_processed = y_data.astype(int).ravel()
        unique_labels = np.unique(y_data_processed)
        print(f"S{subject_id}: Unique labels in dataset: {unique_labels}")
        if len(unique_labels) < 2:
            print(f"S{subject_id}: ERROR - Need at least 2 classes, found {len(unique_labels)}")
            return None, None, None, None
        if not all(label in [1, 2] for label in unique_labels):
            print(f"S{subject_id}: Labels are not in [1, 2] format. Filtering...")
            valid_indices = np.where((y_data_processed == 1) | (y_data_processed == 2))[0]
            if len(valid_indices) < len(y_data_processed):
                print(f"S{subject_id}: Keeping only trials with labels 1 or 2 ({len(valid_indices)}/{len(y_data_processed)} trials).")
                x_data = x_data[:, :, valid_indices]
                y_data_processed = y_data_processed[valid_indices]
            if len(np.unique(y_data_processed)) < 2:
                 print(f"S{subject_id}: ERROR - Not enough valid labels after filtering.")
                 return None, None, None, None
    except Exception as e:
        print(f"S{subject_id}: ERROR processing labels: {e}"); traceback.print_exc(); return None, None, None, None

    # Check class distribution
    try:
        left_indices = np.where(y_data_processed == 1)[0]
        right_indices = np.where(y_data_processed == 2)[0]
        print(f"S{subject_id}: Found {len(left_indices)} trials for class 1 (Left)")
        print(f"S{subject_id}: Found {len(right_indices)} trials for class 2 (Right)")
        if len(left_indices) == 0 or len(right_indices) == 0:
            print(f"S{subject_id}: ERROR - Missing trials for one class"); return None, None, None, None
    except Exception as e:
        print(f"S{subject_id}: ERROR checking class distribution: {e}"); traceback.print_exc(); return None, None, None, None

    # Segment data (extract time window - standard 2s window @ 250Hz = 500 samples)
    # Assuming cue onset is around 1s (sample 250), take 0.5s post-cue to 2.5s post-cue
    # Adjust based on your specific paradigm timing if known.
    # Example: take samples from 115 (0.46s) to 615 (2.46s) -> 500 samples
    segment_start_sample = 115
    segment_end_sample = 615
    if x_data.shape[0] < segment_end_sample:
         print(f"S{subject_id}: WARNING - Data shorter than expected ({x_data.shape[0]} samples). Adjusting segment.")
         segment_end_sample = x_data.shape[0]
         segment_start_sample = max(0, segment_end_sample - 500)
         print(f"S{subject_id}: Using segment {segment_start_sample}:{segment_end_sample}")

    try:
        x_segmented = x_data[segment_start_sample:segment_end_sample, :, :]
        n_timepoints_seg, n_channels_seg, _ = x_segmented.shape
        print(f"S{subject_id}: Segmented data shape: {x_segmented.shape}")
        if n_timepoints_seg < 100: # Min reasonable length
             print(f"S{subject_id}: ERROR - Segmented data too short: {n_timepoints_seg} samples"); return None, None, None, None
    except Exception as e:
        print(f"S{subject_id}: ERROR during data segmentation: {e}"); traceback.print_exc(); return None, None, None, None

    # Filtering
    x_train_filtered_list = []
    valid_trial_indices = []
    try:
        num_trials = x_segmented.shape[2]
        print(f"S{subject_id}: Filtering {num_trials} trials...")
        for i in range(num_trials):
            try:
                trial_data = x_segmented[:, :, i]
                trial_filtered = elliptical_filter(trial_data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS)
                if trial_filtered.shape == trial_data.shape and not np.isnan(trial_filtered).any():
                    x_train_filtered_list.append(trial_filtered)
                    valid_trial_indices.append(i) # Keep track of which trials were successful
                else:
                    print(f"S{subject_id}: WARNING - Filter failed or changed shape for trial {i}. Skipping.")
            except Exception as e:
                print(f"S{subject_id}: WARNING - Filter error on train trial {i}: {e}")
        if not x_train_filtered_list:
             print(f"S{subject_id}: ERROR - No trials survived filtering."); return None, None, None, None
    except Exception as e:
        print(f"S{subject_id}: ERROR during filtering loop: {e}"); traceback.print_exc(); return None, None, None, None

    # Stack, transpose, normalize
    try:
        y_train = y_data_processed[valid_trial_indices] # Filter labels accordingly
        x_train_filtered = np.stack(x_train_filtered_list, axis=-1)
        print(f"S{subject_id}: Filtered data shape: {x_train_filtered.shape}")
        x_train_final_transposed = np.transpose(x_train_filtered, (2, 1, 0)) # -> (trials, channels, timepoints)
        x_train_final = _normalize_trials_internal(x_train_final_transposed)

        N_TRIALS, N_CHANNELS, N_TIMEPOINTS = x_train_final.shape # Get dimensions from final data
        print(f"S{subject_id}: Training data final shape: {x_train_final.shape} with {len(y_train)} labels")
        print(f"S{subject_id}: Determined N_CHANNELS={N_CHANNELS}, N_TIMEPOINTS={N_TIMEPOINTS}")

        return x_train_final, y_train, N_CHANNELS, N_TIMEPOINTS
    except Exception as e:
        print(f"S{subject_id}: ERROR during final data preparation: {e}"); traceback.print_exc(); return None, None, None, None


def generate_synthetic_data(left_generator, right_generator, num_samples_per_class, n_channels, n_timepoints, z_dim=Z_DIM):
    """Generates synthetic data using loaded generator models."""
    if left_generator is None or right_generator is None:
        print("ERROR: One or both generators not loaded."); return None, None
    # Verify generator output shape if possible (might need input spec)
    # print("Generator Left Output Shape:", left_generator.output_shape)
    # print("Generator Right Output Shape:", right_generator.output_shape)

    print(f"Generating {num_samples_per_class} synthetic samples per class...")
    noise_left = tf.random.normal([num_samples_per_class, z_dim])
    noise_right = tf.random.normal([num_samples_per_class, z_dim])

    try:
        synth_left = left_generator(noise_left, training=False)
        synth_right = right_generator(noise_right, training=False)

        # Ensure output matches expected shape (trials, channels, timepoints)
        expected_shape = (num_samples_per_class, n_channels, n_timepoints)
        if synth_left.shape != expected_shape or synth_right.shape != expected_shape:
             print(f"WARNING: Generator output shape mismatch!")
             print(f"Expected: {expected_shape}, Got Left: {synth_left.shape}, Got Right: {synth_right.shape}")
             # Attempt to reshape if dimensions match otherwise
             try:
                 synth_left = tf.reshape(synth_left, expected_shape)
                 synth_right = tf.reshape(synth_right, expected_shape)
                 print("--- Reshaped generator output.")
             except Exception as reshape_err:
                 print(f"--- ERROR: Could not reshape generator output: {reshape_err}. FID might fail.")
                 # Continue cautiously, FID might fail later if shapes are wrong

        synth_data = tf.concat([synth_left, synth_right], axis=0).numpy()
        synth_labels = np.concatenate([np.ones(num_samples_per_class, dtype=int), np.ones(num_samples_per_class, dtype=int) * 2])
        indices = np.arange(synth_data.shape[0]); np.random.shuffle(indices)
        synth_data = synth_data[indices]; synth_labels = synth_labels[indices]
        print(f"Generated synthetic data shape: {synth_data.shape}")

        # Apply filtering and normalization *identically* to how real data was processed
        # Note: Preprocessing during training already included filtering and normalization.
        # For FID/analysis, we should compare *post-processed* real data to *post-processed* synthetic data.
        # If generators were trained on filtered+normalized data, their output should represent that.
        # We may not need explicit filtering/normalization here IF the generators learned it.
        # However, applying it ensures consistency, especially if generation introduces artifacts.

        print("Applying post-generation filtering & normalization to synthetic data...")
        filtered_synth_data_list = []
        valid_synth_indices = []
        for i in range(synth_data.shape[0]):
            # Transpose to (timepoints, channels) for filtering function
            trial_data = np.transpose(synth_data[i], (1, 0)) # (channels, timepoints) -> (timepoints, channels)
            try:
                filtered_trial = elliptical_filter(trial_data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS)
                if filtered_trial.shape == trial_data.shape and not np.isnan(filtered_trial).any():
                     # Transpose back to (channels, timepoints) and store
                     filtered_synth_data_list.append(np.transpose(filtered_trial, (1, 0)))
                     valid_synth_indices.append(i)
                else:
                    print(f"Warning: Filter failed/changed shape for synthetic trial {i}. Skipping.")
            except Exception as filter_e:
                print(f"Warning: Filter error for synthetic trial {i}: {filter_e}. Skipping.")

        if not filtered_synth_data_list:
            print("ERROR: No synthetic trials survived filtering.")
            return None, None

        filtered_synth_data = np.stack(filtered_synth_data_list, axis=0) # (trials, channels, timepoints)
        filtered_synth_labels = synth_labels[valid_synth_indices]

        # Normalize the filtered data trial-by-trial
        normalized_synth_data = _normalize_trials_internal(filtered_synth_data)

        print(f"Filtered & normalized synthetic data shape: {normalized_synth_data.shape}")
        return normalized_synth_data, filtered_synth_labels

    except Exception as e:
        print(f"ERROR generating/processing synthetic data: {e}"); traceback.print_exc(); return None, None

# --- Analysis Function: Grand Average (Time Domain) ---
def compute_grand_average(data, labels, channel_idx, class_label, confidence=CONFIDENCE_LEVEL):
    """Computes grand average for a single class and channel."""
    class_indices = np.where(labels.squeeze() == class_label)[0]
    if len(class_indices) == 0: return None
    class_data = data[class_indices, channel_idx, :] # (n_trials, n_timepoints)
    grand_avg = np.mean(class_data, axis=0)
    sem = stats.sem(class_data, axis=0, nan_policy='omit') # Handle potential NaNs if filter failed
    if np.isnan(sem).any() or len(class_indices) <= 1: # SEM is NaN if only 1 trial or all NaNs
         lower_ci, upper_ci = grand_avg, grand_avg # No meaningful CI
    else:
        ci_factor = stats.t.ppf((1 + confidence) / 2., len(class_indices)-1)
        lower_ci = grand_avg - sem * ci_factor
        upper_ci = grand_avg + sem * ci_factor
    return {"grand_avg": grand_avg, "lower_ci": lower_ci, "upper_ci": upper_ci, "n_trials": len(class_indices), "sem": sem}

def plot_grand_average_comparison(real_results, synth_results, channel_name, class_label,
                                 sampling_rate, output_dir, subject_id):
    """Plots comparison and returns stats for Time Domain Grand Average."""
    if real_results is None or synth_results is None:
        print(f"Skipping Grand Average plot for Class {class_label}, Chan {channel_name} - Missing data.")
        return None
    time_points = np.arange(len(real_results["grand_avg"])) / sampling_rate
    n_real, n_synth = real_results["n_trials"], synth_results["n_trials"]
    plt.figure(figsize=(12, 6))
    # Annotate SNR improvement
    if n_real > 0: plt.annotate(f"SNR Imp. (Real): {np.sqrt(n_real):.2f}x", xy=(0.02, 0.05), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), fontsize=9)
    if n_synth > 0: plt.annotate(f"SNR Imp. (Synth): {np.sqrt(n_synth):.2f}x", xy=(0.02, 0.12), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), fontsize=9)
    # Plot real
    plt.plot(time_points, real_results["grand_avg"], 'b-', linewidth=2, label=f'Real Data (n={n_real})')
    plt.fill_between(time_points, real_results["lower_ci"], real_results["upper_ci"], color='blue', alpha=0.2)
    # Plot synthetic
    plt.plot(time_points, synth_results["grand_avg"], 'r-', linewidth=2, label=f'Synthetic Data (n={n_synth})')
    plt.fill_between(time_points, synth_results["lower_ci"], synth_results["upper_ci"], color='red', alpha=0.2)
    plt.title(f"Subject {subject_id}: Grand Average ERP - Class {class_label}, Channel {channel_name}")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude (Normalized)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"S{subject_id}_GrandAvg_ERP_C{class_label}_Chan{channel_name.replace(' ','')}.png")
    plt.savefig(plot_filename); print(f"--- Saved ERP plot: {plot_filename}"); plt.close()
    # Calculate stats
    try:
        # Ensure no NaNs before calculating stats
        real_avg_clean = real_results["grand_avg"][~np.isnan(real_results["grand_avg"])]
        synth_avg_clean = synth_results["grand_avg"][~np.isnan(synth_results["grand_avg"])]
        if len(real_avg_clean) != len(synth_avg_clean) or len(real_avg_clean) < 2:
             print("ERP Stat Warning: Arrays have different lengths after NaN removal or are too short.")
             corr, rmse, freq_dist = np.nan, np.nan, np.nan
        else:
            corr, _ = stats.pearsonr(real_avg_clean, synth_avg_clean)
            rmse = np.sqrt(np.mean((real_avg_clean - synth_avg_clean)**2))
            real_fft = np.abs(fft(real_avg_clean)); synth_fft = np.abs(fft(synth_avg_clean))
            freq_dist = wasserstein_distance(real_fft[:len(real_fft)//2], synth_fft[:len(synth_fft)//2])
    except ValueError as e:
        print(f"ERP Stat Warning: Calculation failed ({e}). Setting stats to NaN.")
        corr, rmse, freq_dist = np.nan, np.nan, np.nan

    return [{"Analysis": "Grand Average ERP", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "Correlation", "Value": corr},
            {"Analysis": "Grand Average ERP", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "RMSE", "Value": rmse},
            {"Analysis": "Grand Average ERP", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "Freq Domain Wasserstein Dist", "Value": freq_dist}]


# --- Analysis Function: Time-Frequency Analysis ---
def compute_time_frequency_grand_average(data, labels, channel_idx, class_label,
                                        sampling_rate=FS, nperseg=TF_NPERSEG, noverlap=TF_NOVERLAP):
    """Computes grand average time-frequency representation for a single class and channel."""
    class_indices = np.where(labels.squeeze() == class_label)[0]
    if len(class_indices) == 0: return None
    class_data = data[class_indices, channel_idx, :]
    tf_arrays = []
    frequencies, times = None, None
    for trial_idx in range(len(class_indices)):
        trial = class_data[trial_idx]
        if np.isnan(trial).any(): continue # Skip if trial has NaNs
        try:
             f, t, Sxx = spectrogram(trial, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
             if frequencies is None: frequencies, times = f, t
             Sxx = 10 * np.log10(Sxx + 1e-10) # Convert to dB, avoid log(0)
             if not np.isnan(Sxx).any(): tf_arrays.append(Sxx) # Check for NaNs after log10
        except ValueError as e:
             print(f"Spectrogram warning on trial {trial_idx}: {e}")
             continue # Skip trial if spectrogram fails

    if not tf_arrays: return None
    tf_arrays = np.array(tf_arrays)
    grand_avg_tf = np.mean(tf_arrays, axis=0)
    return {"frequencies": frequencies, "times": times, "grand_avg_tf": grand_avg_tf, "n_trials": len(tf_arrays)}

def plot_time_frequency_comparison(real_results, synth_results, channel_name, class_label,
                                  freq_range, output_dir, subject_id):
    """Plots comparison and returns stats for Time-Frequency Grand Average."""
    if real_results is None or synth_results is None: print(f"Skipping TF plot for Class {class_label}, Chan {channel_name} - Missing data."); return None
    if real_results["grand_avg_tf"].shape != synth_results["grand_avg_tf"].shape: print(f"Skipping TF plot for Class {class_label}, Chan {channel_name} - Mismatched shapes."); return None
    real_f, real_t, real_tf = real_results["frequencies"], real_results["times"], real_results["grand_avg_tf"]
    synth_f, synth_t, synth_tf = synth_results["frequencies"], synth_results["times"], synth_results["grand_avg_tf"]
    n_real, n_synth = real_results["n_trials"], synth_results["n_trials"]
    f_mask = (real_f >= freq_range[0]) & (real_f <= freq_range[1])
    masked_f = real_f[f_mask]
    if not np.any(f_mask): print(f"Warning: No frequencies found in {freq_range} for TF plotting."); return None

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    vmin = min(np.min(real_tf[f_mask, :]), np.min(synth_tf[f_mask, :]))
    vmax = max(np.max(real_tf[f_mask, :]), np.max(synth_tf[f_mask, :]))
    # Plot real
    im1 = axs[0].pcolormesh(real_t, masked_f, real_tf[f_mask, :], shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_ylabel('Frequency (Hz)'); axs[0].set_title(f'Real Data - Class {class_label}, Chan {channel_name} (n={n_real})')
    # Plot synthetic
    im2 = axs[1].pcolormesh(synth_t, masked_f, synth_tf[f_mask, :], shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_ylabel('Frequency (Hz)'); axs[1].set_xlabel('Time (s)'); axs[1].set_title(f'Synthetic Data - Class {class_label}, Chan {channel_name} (n={n_synth})')
    fig.colorbar(im2, ax=axs, label='Power (dB)'); plt.suptitle(f"Subject {subject_id}: Time-Frequency Spectrogram Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(output_dir, f"S{subject_id}_TimeFreq_C{class_label}_Chan{channel_name.replace(' ','')}.png")
    plt.savefig(plot_filename); print(f"--- Saved TF plot: {plot_filename}"); plt.close(fig)
    # Plot difference
    plt.figure(figsize=(12, 5))
    diff_tf = real_tf - synth_tf
    diff_min, diff_max = np.min(diff_tf[f_mask, :]), np.max(diff_tf[f_mask, :])
    divnorm = colors.TwoSlopeNorm(vmin=diff_min, vcenter=0., vmax=diff_max)
    im_diff = plt.pcolormesh(real_t, masked_f, diff_tf[f_mask, :], norm=divnorm, cmap='RdBu_r', shading='gouraud')
    plt.colorbar(im_diff, label='Difference (Real - Synthetic) dB'); plt.ylabel('Frequency (Hz)'); plt.xlabel('Time (s)')
    plt.title(f'Subject {subject_id}: Time-Frequency Difference - Class {class_label}, Channel {channel_name}'); plt.tight_layout()
    plot_filename_diff = os.path.join(output_dir, f"S{subject_id}_TimeFreqDiff_C{class_label}_Chan{channel_name.replace(' ','')}.png")
    plt.savefig(plot_filename_diff); print(f"--- Saved TF difference plot: {plot_filename_diff}"); plt.close()
    # Calculate stats
    try:
        real_tf_masked = real_tf[f_mask, :].flatten()
        synth_tf_masked = synth_tf[f_mask, :].flatten()
        corr, _ = stats.pearsonr(real_tf_masked, synth_tf_masked)
        rmse = np.sqrt(np.mean((real_tf_masked - synth_tf_masked)**2))
    except ValueError as e:
        print(f"TF Stat Warning: Calculation failed ({e}). Setting stats to NaN.")
        corr, rmse = np.nan, np.nan
    return [{"Analysis": "Time-Frequency", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "Correlation (TF dB)", "Value": corr},
            {"Analysis": "Time-Frequency", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "RMSE (TF dB)", "Value": rmse}]


# --- Analysis Function: Power Spectral Density (PSD) ---
def compute_average_psd(data, labels, channel_idx, class_label,
                       sampling_rate=FS, nperseg=PSD_NPERSEG, noverlap=PSD_NOVERLAP):
    """Computes average PSD for a single class and channel."""
    class_indices = np.where(labels.squeeze() == class_label)[0]
    if len(class_indices) == 0: return None
    class_data = data[class_indices, channel_idx, :]
    all_psds = []
    frequencies = None
    for i in range(len(class_indices)):
        trial = class_data[i]
        if np.isnan(trial).any(): continue # Skip if trial has NaNs
        try:
            f, psd = welch(trial, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
            if frequencies is None: frequencies = f
            if not np.isnan(psd).any(): all_psds.append(psd)
        except ValueError as e:
            print(f"Welch warning on trial {i}: {e}")
            continue # Skip trial if welch fails

    if not all_psds: return None
    all_psds = np.array(all_psds)
    avg_psd = np.mean(all_psds, axis=0)
    sem_psd = stats.sem(all_psds, axis=0, nan_policy='omit')
    return {"frequencies": frequencies, "avg_psd": avg_psd, "sem_psd": sem_psd, "all_psds": all_psds, "n_trials": len(all_psds)}

def plot_psd_comparison(real_results, synth_results, channel_name, class_label,
                       freq_range, output_dir, subject_id, bands):
    """Plots PSD comparison, saves plot, and calculates related stats."""
    if real_results is None or synth_results is None: print(f"Skipping PSD plot for Class {class_label}, Chan {channel_name} - Missing data."); return None
    real_f, real_psd, real_sem = real_results["frequencies"], real_results["avg_psd"], real_results["sem_psd"]
    synth_f, synth_psd, synth_sem = synth_results["frequencies"], synth_results["avg_psd"], synth_results["sem_psd"]
    n_real, n_synth = real_results["n_trials"], synth_results["n_trials"]
    freq_mask_real = (real_f >= freq_range[0]) & (real_f <= freq_range[1])
    freq_mask_synth = (synth_f >= freq_range[0]) & (synth_f <= freq_range[1])
    if not np.any(freq_mask_real) or not np.any(freq_mask_synth): print(f"Warning: PSD range {freq_range} has no data."); return None

    plt.figure(figsize=(12, 6))
    ci_factor = 1.96 # 95% CI
    # Plot real PSD
    plt.plot(real_f[freq_mask_real], real_psd[freq_mask_real], 'b-', linewidth=2, label=f'Real Data (n={n_real})')
    plt.fill_between(real_f[freq_mask_real], real_psd[freq_mask_real] - ci_factor * real_sem[freq_mask_real], real_psd[freq_mask_real] + ci_factor * real_sem[freq_mask_real], color='blue', alpha=0.2)
    # Plot synth PSD
    plt.plot(synth_f[freq_mask_synth], synth_psd[freq_mask_synth], 'r-', linewidth=2, label=f'Synthetic Data (n={n_synth})')
    plt.fill_between(synth_f[freq_mask_synth], synth_psd[freq_mask_synth] - ci_factor * synth_sem[freq_mask_synth], synth_psd[freq_mask_synth] + ci_factor * synth_sem[freq_mask_synth], color='red', alpha=0.2)
    plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD (Normalized²/Hz)'); plt.title(f'Subject {subject_id}: PSD - Class {class_label}, Channel {channel_name}')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.xlim(freq_range); plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"S{subject_id}_PSD_C{class_label}_Chan{channel_name.replace(' ','')}.png")
    plt.savefig(plot_filename); print(f"--- Saved PSD plot: {plot_filename}"); plt.close()
    # --- Calculate Statistics ---
    stats_list = []
    # 1. Overall PSD Similarity
    try:
        if np.array_equal(real_f[freq_mask_real], synth_f[freq_mask_synth]):
            real_psd_masked = real_psd[freq_mask_real]
            synth_psd_masked = synth_psd[freq_mask_synth]
            # Check for NaNs before calculation
            valid_idx = ~np.isnan(real_psd_masked) & ~np.isnan(synth_psd_masked)
            if np.sum(valid_idx) > 1: # Need at least 2 points for correlation
                 corr_psd, _ = stats.pearsonr(real_psd_masked[valid_idx], synth_psd_masked[valid_idx])
                 rmse_psd = np.sqrt(np.mean((real_psd_masked[valid_idx] - synth_psd_masked[valid_idx])**2))
            else:
                 corr_psd, rmse_psd = np.nan, np.nan
            stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD Correlation", "Value": corr_psd})
            stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD RMSE", "Value": rmse_psd})
        else:
            print(f"Warning: Freq bins mismatch. Skipping overall PSD correlation/RMSE.")
            stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD Correlation", "Value": np.nan})
            stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD RMSE", "Value": np.nan})
    except ValueError as e:
        print(f"PSD Stat Warning: Calculation failed ({e}). Setting stats to NaN.")
        stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD Correlation", "Value": np.nan})
        stats_list.append({"Analysis": "PSD", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": "PSD RMSE", "Value": np.nan})

    # 2. Band Power and Distribution Analysis
    for band_name, (low_freq, high_freq) in bands.items():
        try:
            band_mask_real = (real_f >= low_freq) & (real_f <= high_freq)
            band_mask_synth = (synth_f >= low_freq) & (synth_f <= high_freq)
            avg_real_band_power, avg_synth_band_power = np.nan, np.nan
            if np.any(band_mask_real): avg_real_band_power = np.mean(real_results["avg_psd"][band_mask_real])
            if np.any(band_mask_synth): avg_synth_band_power = np.mean(synth_results["avg_psd"][band_mask_synth])
            stats_list.append({"Analysis": "PSD Band Power", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Power Diff (Real-Synth)", "Value": avg_real_band_power - avg_synth_band_power})
            stats_list.append({"Analysis": "PSD Band Power", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Power Real", "Value": avg_real_band_power})
            stats_list.append({"Analysis": "PSD Band Power", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Power Synth", "Value": avg_synth_band_power})

            # Band Power Distribution Comparison
            real_trial_band_powers = np.mean(real_results["all_psds"][:, band_mask_real], axis=1) if np.any(band_mask_real) else np.array([np.nan])
            synth_trial_band_powers = np.mean(synth_results["all_psds"][:, band_mask_synth], axis=1) if np.any(band_mask_synth) else np.array([np.nan])
            # Remove NaNs before stat tests
            real_trial_band_powers = real_trial_band_powers[~np.isnan(real_trial_band_powers)]
            synth_trial_band_powers = synth_trial_band_powers[~np.isnan(synth_trial_band_powers)]

            ks_stat, p_value, w_distance = np.nan, np.nan, np.nan
            if len(real_trial_band_powers) > 1 and len(synth_trial_band_powers) > 1:
                 ks_stat, p_value = ks_2samp(real_trial_band_powers, synth_trial_band_powers)
                 w_distance = wasserstein_distance(real_trial_band_powers, synth_trial_band_powers)

            stats_list.append({"Analysis": "PSD Band Distribution", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} KS p-value", "Value": p_value})
            stats_list.append({"Analysis": "PSD Band Distribution", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Wasserstein Dist", "Value": w_distance})
        except Exception as band_e:
             print(f"Error calculating stats for band {band_name}: {band_e}")
             # Add NaN entries if calculation failed
             stats_list.append({"Analysis": "PSD Band Power", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Power Diff (Real-Synth)", "Value": np.nan})
             # ... add NaNs for other metrics in this band ...
             stats_list.append({"Analysis": "PSD Band Distribution", "Subject": subject_id, "Channel": channel_name, "Class": class_label, "Metric": f"{band_name.capitalize()} Wasserstein Dist", "Value": np.nan})


    return stats_list

# --- FID Calculation Functions ---

def calculate_fid(model, real_images, fake_images):
    """Calculates the FID score using activations from a pre-trained model."""
    # Note: InceptionV3's preprocess_input scales images from [0, 255] or [-1, 1] to [-1, 1].
    # Ensure our prepared images are in a compatible range (e.g., [0, 1] or [-1, 1]) BEFORE this function.
    print(f"Calculating FID: Real images={real_images.shape}, Fake images={fake_images.shape}")
    if real_images.shape[1:] != fake_images.shape[1:]:
        raise ValueError(f"Image shapes mismatch for FID: {real_images.shape} vs {fake_images.shape}")
    if real_images.shape[-1] != 3:
         raise ValueError(f"Images must have 3 channels for InceptionV3, got {real_images.shape[-1]}")

    # Preprocess images for InceptionV3
    real_images_processed = preprocess_input(real_images * 255.0) # Scale [0,1] to [0,255] then preprocess
    fake_images_processed = preprocess_input(fake_images * 255.0) # Scale [0,1] to [0,255] then preprocess

    # Calculate activations
    act_real = model.predict(real_images_processed, verbose=0)
    act_fake = model.predict(fake_images_processed, verbose=0)
    print(f"Activations shape: Real={act_real.shape}, Fake={act_fake.shape}")

    # Calculate mean and covariance statistics
    mu_real, sigma_real = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(act_fake, axis=0), np.cov(act_fake, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu_real - mu_fake)**2.0)

    # Calculate sqrt of product of cov matrices
    try:
        # Add epsilon to diagonals for numerical stability
        sigma_real += np.eye(sigma_real.shape[0]) * FID_EPS
        sigma_fake += np.eye(sigma_fake.shape[0]) * FID_EPS
        covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False) # Faster & more stable than direct sqrtm(sigma_real @ sigma_fake)
    except Exception as e:
        print(f"ERROR calculating sqrtm of covariance matrices: {e}")
        print("Covariance matrices might be singular. Check input data variance.")
        # Attempt to use pseudo-inverse or return NaN
        return np.nan


    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        print("Warning: Complex numbers generated in FID sqrtm. Taking real part.")
        covmean = covmean.real

    # Calculate score
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    print(f"FID Calculation: ssdiff={ssdiff:.2f}, trace_term={np.trace(sigma_real + sigma_fake - 2.0 * covmean):.2f}, FID={fid:.3f}")
    return fid

def prepare_eeg_for_fid(eeg_data, target_size, fs=FS, nperseg=FID_NPERSEG, noverlap=FID_NOVERLAP):
    """Converts EEG trials (n_trials, n_timepoints) into spectrogram images for FID."""
    images = []
    print(f"Preparing {eeg_data.shape[0]} EEG trials for FID...")
    if eeg_data.ndim != 2:
        raise ValueError(f"Expected eeg_data to be 2D (trials, timepoints), got {eeg_data.ndim}D")

    for i in range(eeg_data.shape[0]):
        trial = eeg_data[i]
        if np.isnan(trial).any(): continue # Skip trials with NaNs

        try:
            # 1. Compute Spectrogram
            f, t, Sxx = spectrogram(trial, fs=fs, nperseg=nperseg, noverlap=noverlap)
            # Use magnitude or power, log scale is common
            Sxx_db = 10 * np.log10(Sxx + 1e-10) # Convert to dB, add epsilon

            # 2. Resize Spectrogram (Frequency x Time)
            # Use tf.image.resize for consistency with TF model
            # Add channel dim for resize, then remove
            Sxx_db_resized = tf.image.resize(Sxx_db[..., tf.newaxis], target_size, method='bilinear')
            Sxx_db_resized = tf.squeeze(Sxx_db_resized, axis=-1).numpy() # Back to 2D

            # 3. Normalize to [0, 1] range (per image normalization)
            min_val, max_val = np.min(Sxx_db_resized), np.max(Sxx_db_resized)
            if max_val - min_val > 1e-6:
                img_normalized = (Sxx_db_resized - min_val) / (max_val - min_val)
            else:
                img_normalized = np.zeros(target_size) # Avoid division by zero

            # 4. Convert to 3 Channels (RGB) by repeating the channel
            img_rgb = np.stack([img_normalized]*3, axis=-1) # Shape (height, width, 3)

            images.append(img_rgb)

        except ValueError as e:
            print(f"Spectrogram/Resize warning on FID prep trial {i}: {e}")
            continue # Skip trial if spectrogram or resize fails
        except Exception as e:
            print(f"Unexpected error preparing FID image for trial {i}: {e}")
            continue

    if not images:
        print("ERROR: No images were successfully prepared for FID.")
        return None

    return np.array(images, dtype=np.float32) # Ensure float32 for TF model


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Analysis Script (with FID)...")

    # Find the specific run directory for the subject
    run_dir = None
    for d in os.listdir(BASE_OUTPUT_DIR):
         if d.startswith('run_') and os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d)):
              potential_subject_dir = os.path.join(BASE_OUTPUT_DIR, d, f"subject_{SUBJECT_ID_TO_ANALYZE}")
              if os.path.exists(potential_subject_dir):
                    run_dir = os.path.join(BASE_OUTPUT_DIR, d)
                    subject_dir = potential_subject_dir
                    print(f"Found subject data in run: {d}")
                    break # Found the correct run and subject dir

    if run_dir is None or subject_dir is None:
        print(f"ERROR: No run directory containing subject_{SUBJECT_ID_TO_ANALYZE} found in {BASE_OUTPUT_DIR}")
        exit()

    all_stats_results = [] # List to store dictionaries of stats

    # 1. Load Real Training Data
    print(f"\nLoading Real Training Data for Subject {SUBJECT_ID_TO_ANALYZE}...")
    x_train_real, y_train_real, N_CHANNELS, N_TIMEPOINTS = None, None, None, None
    try:
        # Adjust path as needed
        mat_file_path = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', 'data1.mat')
        if not os.path.exists(mat_file_path):
            # Try looking one level up if running from within BASE_OUTPUT_DIR
             alt_path = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '..', 'data1.mat')
             if os.path.exists(alt_path):
                 mat_file_path = alt_path
             else:
                raise FileNotFoundError(f"data1.mat not found in current dir or parent dir.")

        data_train = scipy.io.loadmat(mat_file_path)
        print("MAT file keys:", list(data_train.keys()))
        
        # Adapt based on actual MAT file structure (common variation: 'data' or 'EEG')
        if 'xsubi_all' in data_train:
             subjects_struct_train = data_train['xsubi_all'][0]
        elif 'data' in data_train and isinstance(data_train['data'], np.ndarray): # Another common format
             subjects_struct_train = data_train['data'][0]
        else:
            raise KeyError("Required data structure ('xsubi_all' or similar) not found in the .mat file")

        if SUBJECT_ID_TO_ANALYZE <= 0 or SUBJECT_ID_TO_ANALYZE > len(subjects_struct_train):
            raise ValueError(f"Subject ID {SUBJECT_ID_TO_ANALYZE} out of range (1 to {len(subjects_struct_train)})")

        subject_data_train = subjects_struct_train[SUBJECT_ID_TO_ANALYZE - 1] # 0-based index
        
        # Determine keys within the subject structure (can be 'X', 'Y' or 'x', 'y' etc.)
        data_key = 'x' if 'x' in subject_data_train.dtype.names else 'X' if 'X' in subject_data_train.dtype.names else None
        label_key = 'y' if 'y' in subject_data_train.dtype.names else 'Y' if 'Y' in subject_data_train.dtype.names else None
        
        if not data_key or not label_key:
            raise KeyError(f"Could not find data ('x' or 'X') or label ('y' or 'Y') keys in subject struct. Keys: {subject_data_train.dtype.names}")
            
        print(f"Using data key '{data_key}' and label key '{label_key}' for S{SUBJECT_ID_TO_ANALYZE}")

        x_train_real, y_train_real, N_CHANNELS, N_TIMEPOINTS = preprocess_training_data(
            subject_data_train[data_key], subject_data_train[label_key], SUBJECT_ID_TO_ANALYZE
        )
        if x_train_real is None:
            raise ValueError("Preprocessing of real data failed.")
        print(f"--- Real data loaded and preprocessed successfully: {x_train_real.shape}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit()
    except Exception as e:
        print(f"ERROR loading or preprocessing real training data: {e}")
        traceback.print_exc()
        exit()


    # 2. Load Saved Generator Models
    print(f"\nLoading Generator Models for Subject {SUBJECT_ID_TO_ANALYZE}...")
    generator_left, generator_right = None, None
    # Check for both .h5 and .keras extensions for flexibility
    left_model_path = None
    right_model_path = None
    
    potential_left_paths = [os.path.join(subject_dir, f"best_left_generator_run2.h5")]
    potential_right_paths = [os.path.join(subject_dir, f"best_right_generator_run2.h5")]
                             
    for p in potential_left_paths:
        if os.path.exists(p): left_model_path = p; break
    for p in potential_right_paths:
        if os.path.exists(p): right_model_path = p; break
        
    if not left_model_path or not right_model_path:
        print(f"ERROR: Generator model file(s) not found in {subject_dir}")
        print(f"Looked for: {potential_left_paths} and {potential_right_paths}")
        exit()
        
    try:
        print(f"Loading left generator from: {left_model_path}")
        generator_left = tf.keras.models.load_model(left_model_path, compile=False) # compile=False speeds up loading if not retraining
        print(f"Loading right generator from: {right_model_path}")
        generator_right = tf.keras.models.load_model(right_model_path, compile=False)
        print("--- Generator models loaded successfully.")
        # generator_left.summary() # Optional
    except Exception as e:
        print(f"ERROR loading generator models: {e}"); traceback.print_exc()
        exit()

    # 3. Generate Synthetic Data
    # Pass N_CHANNELS and N_TIMEPOINTS determined from real data to ensure consistency
    x_synth, y_synth = generate_synthetic_data(generator_left, generator_right, N_SYNTH_SAMPLES_PER_CLASS, N_CHANNELS, N_TIMEPOINTS, Z_DIM)
    if x_synth is None: print("ERROR: Failed to generate synthetic data."); exit()

    # --- Load InceptionV3 model for FID ---
    print("\nLoading InceptionV3 model for FID...")
    try:
        inception_model = InceptionV3(include_top=False, pooling='avg', weights='imagenet',
                                       input_shape=(FID_IMAGE_SIZE[0], FID_IMAGE_SIZE[1], 3))
        print("--- InceptionV3 model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading InceptionV3 model: {e}")
        print("Ensure you have internet connection for first download or have weights cached.")
        traceback.print_exc()
        inception_model = None # Set to None so FID calculation is skipped

    # 4. Perform Analyses Channel by Channel, Class by Class
    for channel_name, channel_idx in CHANNELS_TO_ANALYZE.items():
        print(f"\n{'='*20} Analyzing Channel: {channel_name} (Index: {channel_idx}) {'='*20}")
        if channel_idx >= N_CHANNELS:
             print(f"WARNING: Channel index {channel_idx} is out of bounds for loaded data ({N_CHANNELS} channels). Skipping channel {channel_name}.")
             continue

        for class_label in [1, 2]: # Class 1 (Left Hand), Class 2 (Right Hand)
            print(f"\n--- Processing Class {class_label} ---")

            # Get indices for the current class
            real_class_indices = np.where(y_train_real.squeeze() == class_label)[0]
            synth_class_indices = np.where(y_synth.squeeze() == class_label)[0]

            # --- Grand Average ERP Analysis ---
            print("Computing Grand Average ERPs...")
            real_erp_results = compute_grand_average(x_train_real, y_train_real, channel_idx, class_label)
            synth_erp_results = compute_grand_average(x_synth, y_synth, channel_idx, class_label)
            erp_stats = plot_grand_average_comparison(real_erp_results, synth_erp_results, channel_name,
                                                      class_label, FS, ANALYSIS_OUTPUT_DIR, SUBJECT_ID_TO_ANALYZE)
            if erp_stats: all_stats_results.extend(erp_stats)

            # --- Time-Frequency Analysis ---
            print("\nComputing Time-Frequency Representations...")
            real_tf_results = compute_time_frequency_grand_average(x_train_real, y_train_real, channel_idx, class_label)
            synth_tf_results = compute_time_frequency_grand_average(x_synth, y_synth, channel_idx, class_label)
            tf_stats = plot_time_frequency_comparison(real_tf_results, synth_tf_results, channel_name,
                                                      class_label, TF_FREQ_RANGE, ANALYSIS_OUTPUT_DIR, SUBJECT_ID_TO_ANALYZE)
            if tf_stats: all_stats_results.extend(tf_stats)

            # --- PSD Analysis ---
            print("\nComputing Power Spectral Densities...")
            real_psd_results = compute_average_psd(x_train_real, y_train_real, channel_idx, class_label)
            synth_psd_results = compute_average_psd(x_synth, y_synth, channel_idx, class_label)
            psd_stats = plot_psd_comparison(real_psd_results, synth_psd_results, channel_name,
                                            class_label, PSD_FREQ_RANGE, ANALYSIS_OUTPUT_DIR, SUBJECT_ID_TO_ANALYZE, BANDS)
            if psd_stats: all_stats_results.extend(psd_stats)

            # --- FID Calculation ---
            print("\nPreparing data and calculating FID score...")
            fid_score = np.nan # Default to NaN
            if inception_model is not None: # Only proceed if Inception model loaded
                 if len(real_class_indices) > 1 and len(synth_class_indices) > 1: # Need >1 sample for covariance
                      try:
                          # Extract data for the current channel and class (trials, timepoints)
                          real_eeg_fid_input = x_train_real[real_class_indices, channel_idx, :]
                          synth_eeg_fid_input = x_synth[synth_class_indices, channel_idx, :]

                          # Prepare images (spectrograms)
                          real_fid_images = prepare_eeg_for_fid(real_eeg_fid_input, FID_IMAGE_SIZE)
                          synth_fid_images = prepare_eeg_for_fid(synth_eeg_fid_input, FID_IMAGE_SIZE)

                          # Calculate FID if images were prepared successfully
                          if real_fid_images is not None and synth_fid_images is not None:
                               if real_fid_images.shape[0] > 1 and synth_fid_images.shape[0] > 1: # Check again after prep
                                    fid_score = calculate_fid(inception_model, real_fid_images, synth_fid_images)
                                    print(f"--- FID Score (Class {class_label}, Chan {channel_name}): {fid_score:.3f}")
                               else:
                                     print("--- Skipping FID: Not enough valid images prepared for covariance calculation.")
                          else:
                              print("--- Skipping FID: Image preparation failed.")

                      except Exception as fid_e:
                          print(f"ERROR calculating FID for Class {class_label}, Chan {channel_name}: {fid_e}")
                          traceback.print_exc()
                          fid_score = np.nan # Ensure it's NaN on error
                 else:
                     print(f"--- Skipping FID: Not enough samples for Class {class_label} (Real: {len(real_class_indices)}, Synth: {len(synth_class_indices)})")
            else:
                print("--- Skipping FID: InceptionV3 model not loaded.")

            # Add FID score to results
            all_stats_results.append({"Analysis": "FID", "Subject": SUBJECT_ID_TO_ANALYZE, "Channel": channel_name,
                                      "Class": class_label, "Metric": "FID Score", "Value": fid_score})


    # 5. Save Statistics to CSV
    if all_stats_results:
        stats_df = pd.DataFrame(all_stats_results)
        # Sort for better readability
        stats_df = stats_df.sort_values(by=["Analysis", "Channel", "Class", "Metric"])
        csv_filename = os.path.join(ANALYSIS_OUTPUT_DIR, f"S{SUBJECT_ID_TO_ANALYZE}_analysis_stats_with_FID.csv")
        try:
            stats_df.to_csv(csv_filename, index=False, float_format='%.5f') # Format float precision
            print(f"\n{'='*60}")
            print(f"Statistical results saved to: {csv_filename}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\nERROR saving statistics to CSV: {e}")
    else:
        print("\nNo statistical results were generated to save.")

    print("\nAnalysis Script Finished.")