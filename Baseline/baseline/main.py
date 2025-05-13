import scipy.io
import numpy as np
from scipy.signal import ellipord, ellip, filtfilt
import numpy as np
from scipy.linalg import eigh
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Bandpass filter definition using elliptical filter
def elliptical_filter(data, lowcut=8, highcut=35, fs=250, rp=1, rs=40):
    nyq = 0.5 * fs
    wp = [lowcut / nyq, highcut / nyq]  # Passband
    ws = [(lowcut - 1) / nyq, (highcut + 1) / nyq]  # Stopband
    n, wn = ellipord(wp, ws, rp, rs)  # Filter order and natural frequency
    b, a = ellip(n, rp, rs, wn, btype='band')  # Coefficients
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


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

def preprocess_and_split_data(xsubi1, subject_id='N/A', 
                             r=10, v=1, 
                             lowfreq=8, highfreq=35, fs=250,
                             startSample=105, endSample=635):
    """
    Preprocesses EEG data following the exact logic in the example but adds validation set and normalization.
    
    Args:
        xsubi1: Subject data with 'Right' and 'Left' fields
        subject_id: Subject identifier (for logging)
        r: Number of trials per class for training
        v: Number of trials per class for validation
        lowfreq, highfreq: Bandpass filter parameters
        fs: Sampling frequency
        startSample, endSample: Time window indices
        
    Returns:
        tuple: (finaltrn, finalval, finaltest, train_mean, train_std) or (None, None, None, None, None) on error
    """
    try:
        # --- Check Expected Structure ---
        if 'Right' not in xsubi1.dtype.names or 'Left' not in xsubi1.dtype.names:
            print(f"  ERROR S{subject_id}: Data missing 'Right' or 'Left' fields. Skipping.")
            return None, None, None, None, None

        nTrialsRight = xsubi1['Right'].shape[2]
        nTrialsLeft = xsubi1['Left'].shape[2]
        print(f"  Found {nTrialsRight} Right trials, {nTrialsLeft} Left trials.")

        # --- Check Trial Adequacy ---
        # Need at least r+v trials for training+validation and 1 more for testing
        if nTrialsRight <= r+v or nTrialsLeft <= r+v:
            print(f"  WARNING S{subject_id}: Not enough trials (R:{nTrialsRight}, L:{nTrialsLeft}) to select r={r} train + v={v} validation and have remaining for testing. Skipping.")
            return None, None, None, None, None

        # --- Split Indices ---
        all_right_indices = np.arange(nTrialsRight)
        all_left_indices = np.arange(nTrialsLeft)

        # Randomly select training indices
        selectedTrainRightIdx = np.random.choice(all_right_indices, r, replace=False)
        selectedTrainLeftIdx = np.random.choice(all_left_indices, r, replace=False)
        
        # Remove training indices from available pools
        remaining_right_indices = np.setdiff1d(all_right_indices, selectedTrainRightIdx, assume_unique=True)
        remaining_left_indices = np.setdiff1d(all_left_indices, selectedTrainLeftIdx, assume_unique=True)
        
        # Randomly select validation indices from remaining
        selectedValRightIdx = np.random.choice(remaining_right_indices, v, replace=False)
        selectedValLeftIdx = np.random.choice(remaining_left_indices, v, replace=False)
        
        # Test indices are what remains after taking train and val
        testRightIdx = np.setdiff1d(remaining_right_indices, selectedValRightIdx, assume_unique=True)
        testLeftIdx = np.setdiff1d(remaining_left_indices, selectedValLeftIdx, assume_unique=True)

        print(f"  Train indices: R={len(selectedTrainRightIdx)}, L={len(selectedTrainLeftIdx)}")
        print(f"  Val indices: R={len(selectedValRightIdx)}, L={len(selectedValLeftIdx)}")
        print(f"  Test indices: R={len(testRightIdx)}, L={len(testLeftIdx)}")

        # --- Create Training, Validation and Test Data ---
        trainX_right = xsubi1['Right'][:, :, selectedTrainRightIdx]
        trainX_left = xsubi1['Left'][:, :, selectedTrainLeftIdx]
        
        valX_right = xsubi1['Right'][:, :, selectedValRightIdx]
        valX_left = xsubi1['Left'][:, :, selectedValLeftIdx]
        
        testX_right = xsubi1['Right'][:, :, testRightIdx]
        testX_left = xsubi1['Left'][:, :, testLeftIdx]

        xsubi = {'x': np.concatenate((trainX_right, trainX_left), axis=2),
                 'y': np.concatenate((np.ones(r), np.ones(r) + 1))} # Labels 1 and 2
                 
        vxsubi = {'x': np.concatenate((valX_right, valX_left), axis=2),
                  'y': np.concatenate((np.ones(v), np.ones(v) + 1))} # Labels 1 and 2

        txsubi = {'x': np.concatenate((testX_right, testX_left), axis=2),
                  'y': np.concatenate((np.ones(len(testRightIdx)), np.ones(len(testLeftIdx)) + 1))}

        print(f"  Train data shape: {xsubi['x'].shape}, Val data shape: {vxsubi['x'].shape}, Test data shape: {txsubi['x'].shape}")

        # --- Preprocessing ---
        print("  Applying Elliptical Filter & Time Window...")
        finaltrn1 = {'x': elliptical_filter(xsubi['x'], lowcut=lowfreq, highcut=highfreq, fs=fs)}
        finalval1 = {'x': elliptical_filter(vxsubi['x'], lowcut=lowfreq, highcut=highfreq, fs=fs)}
        finaltest1 = {'x': elliptical_filter(txsubi['x'], lowcut=lowfreq, highcut=highfreq, fs=fs)}

        finaltrn = {'x': finaltrn1['x'][startSample:endSample+1, :, :], 'y': xsubi['y']}
        finalval = {'x': finalval1['x'][startSample:endSample+1, :, :], 'y': vxsubi['y']}
        finaltest = {'x': finaltest1['x'][startSample:endSample+1, :, :], 'y': txsubi['y'],
                    'lowfreq': lowfreq, 'highfreq': highfreq}
        
        print(f"  Processed train shape: {finaltrn['x'].shape}, Processed val shape: {finalval['x'].shape}, Processed test shape: {finaltest['x'].shape}")
        
        # --- Add Z-Score Normalization (based ONLY on Training data) ---
        print("  Calculating Z-score stats based ONLY on the Training data...")
        # Calculate mean and std across samples and trials (keeping channel dimension)
        train_data = finaltrn['x']
        train_mean = np.mean(train_data, axis=(0, 2), keepdims=True)  # Mean per channel
        train_std = np.std(train_data, axis=(0, 2), keepdims=True)    # Std per channel
        train_std[train_std < 1e-8] = 1e-8  # Prevent division by zero
        
        # Apply normalization
        finaltrn['x'] = (finaltrn['x'] - train_mean) / train_std
        finalval['x'] = (finalval['x'] - train_mean) / train_std  # Use train stats
        finaltest['x'] = (finaltest['x'] - train_mean) / train_std  # Use train stats
        
        print("  Applied Z-score normalization (using Train stats) to Train, Val, and Test sets.")
        
        return finaltrn, finalval, finaltest, train_mean, train_std
        
    except Exception as e:
        print(f"ERROR S{subject_id} during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

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

import traceback

# --- Parameters ---
mat_file = "data1.mat"
r = 40    # Fixed number of trials per class for training
v = 10     # Fixed number of trials per class for validation
lowfreq = 8
highfreq = 35
fs = 250  # Sampling frequency
startSample = 115 # 0-indexed
endSample = 615   # 0-indexed
numchannel = 22
seed_value = 42  # For reproducibility across runs

# --- Set Seed (once before the loop) ---
np.random.seed(seed_value)
print(f"Using random seed: {seed_value}")

# --- Load Data ---
try:
    data = sio.loadmat(mat_file)
    xsubi_all = data['xsubi_all']
    print(f"Data loaded successfully from {mat_file}")
except FileNotFoundError:
    print(f"Error: Data file '{mat_file}' not found.")
    exit()
except KeyError:
    print(f"Error: Variable 'xsubi_all' not found in '{mat_file}'. Check the file structure.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the data: {e}")
    exit()

# --- Determine Number of Subjects ---
num_subjects = xsubi_all.shape[1]
print(f"Found {num_subjects} subjects in the data.")

# --- Initialize Results Storage ---
subject_accuracies = np.full(num_subjects, np.nan) # Use NaN for unsuccessful subjects

# --- Loop Through Subjects ---
for subi in range(num_subjects):
    print(f"\n{'='*10} Processing Subject {subi+1}/{num_subjects} {'='*10}")

    try:
        # --- Extract Subject Data ---
        xsubi1 = xsubi_all[0, subi]
        
        # --- Use the new preprocessing and split function ---
        finaltrn, finalval, finaltest, train_mean, train_std = preprocess_and_split_data(
            xsubi1, subject_id=subi+1, r=r, v=v, 
            lowfreq=lowfreq, highfreq=highfreq, fs=fs,
            startSample=startSample, endSample=endSample
        )
        
        # Check if preprocessing was successful
        if finaltrn is None:
            print(f"  ERROR S{subi+1}: Preprocessing failed. Skipping subject.")
            continue
            
        # --- Train and Evaluate using new functions ---
        print("  Applying Common Spatial Patterns (CSP)...")
        model, spatial_filters = train_cspsvm(finaltrn)
        
    
        print("  Training and Evaluating SVM...")
        accuracy, confusion_mat, predictions = evaluate_cspsvm(model, spatial_filters, finaltest)
        
        # Store the result
        subject_accuracies[subi] = accuracy
        print(f"  Subject {subi+1} Test Accuracy: {accuracy:.2f}%")

    # --- Error Handling within the loop ---
    except NameError as e:
        print(f"  ERROR S{subi+1}: A function is likely not defined (NameError). Details: {e}")
        print("  Check if all functions are imported/defined.")
        # Keep default NaN accuracy and continue to next subject
    except ValueError as e:
        # Catch potential dimension mismatches, etc.
        print(f"  ERROR S{subi+1}: Data shape or value issue (ValueError). Details: {e}")
        traceback.print_exc() # Print full error details
        # Keep default NaN accuracy and continue
    except Exception as e:
        # Catch any other unexpected error for this subject
        print(f"  UNEXPECTED ERROR S{subi+1}: {type(e).__name__} occurred. Details: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        # Keep default NaN accuracy and continue

# --- Summarize Results ---
print(f"\n{'='*10} Processing Complete {'='*10}")
print("Accuracies per Subject:")
for i, acc in enumerate(subject_accuracies):
    if np.isnan(acc):
        print(f"  Subject {i+1}: Failed/Skipped")
    else:
        print(f"  Subject {i+1}: {acc:.2f}%")

# Calculate average excluding NaNs
valid_accuracies = subject_accuracies[~np.isnan(subject_accuracies)]
if len(valid_accuracies) > 0:
    average_accuracy = np.mean(valid_accuracies)
    std_dev_accuracy = np.std(valid_accuracies)
    print(f"\nAverage Accuracy across {len(valid_accuracies)} successful subjects: {average_accuracy:.2f}%")
    print(f"Standard Deviation across {len(valid_accuracies)} successful subjects: {std_dev_accuracy:.2f}%")
else:
    print("\nNo subjects processed successfully.")
