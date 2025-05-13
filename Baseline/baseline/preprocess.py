import numpy as np
from scipy.signal import ellip, ellipord, filtfilt

def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    """
    Applies an elliptical bandpass filter to the data.
    """
    nyq = 0.5 * fs
    wp = [lowcut / nyq, highcut / nyq]
    ws = [(lowcut - 1) / nyq, (highcut + 1) / nyq]
    n, wn = ellipord(wp, ws, rp, rs)
    b, a = ellip(n, rp, rs, wn, btype='band')
    return filtfilt(b, a, data, axis=0)

def preprocess_data_matlab_style(x_data, y_data, lowcut=8, highcut=35, fs=250, train_trials=25):
    """
    Preprocesses EEG data using MATLAB-style logic with filtering and splitting.
    """
    right_data = x_data[:, :, y_data.ravel() == 2]
    left_data = x_data[:, :, y_data.ravel() == 1]

    x_train = np.concatenate([right_data[:, :, :train_trials], left_data[:, :, :train_trials]], axis=2)
    y_train = np.concatenate([np.zeros(train_trials) + 2, np.zeros(train_trials) + 1])

    x_test = np.concatenate([right_data[:, :, train_trials:], left_data[:, :, train_trials:]], axis=2)
    y_test = np.concatenate([np.zeros(right_data.shape[2] - train_trials) + 2, np.zeros(left_data.shape[2] - train_trials) + 1])

    x_train_filtered = np.array([elliptical_filter(trial, lowcut, highcut, fs) for trial in np.transpose(x_train, (2, 0, 1))])
    x_test_filtered = np.array([elliptical_filter(trial, lowcut, highcut, fs) for trial in np.transpose(x_test, (2, 0, 1))])

    return (
        np.transpose(x_train_filtered, (0, 2, 1)),
        y_train,
        np.transpose(x_test_filtered, (0, 2, 1)),
        y_test
    )
