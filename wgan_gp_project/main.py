import scipy.io
import os
from src.preprocessing import preprocess_data
from src.training import train_wgan_gp
from src.Training import Train_wgan_gp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

# Get the path to the current script's directory
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'data.mat')  # Reference data.mat in the same directory

    # Load the .mat file
data = scipy.io.loadmat(file_path)

# Access the nested 'data' key
subjects = data['data'][0]  # Array of structures for each subject
subject = subjects[5]

# Extract the data for the current subject
x_data = subject['x']  # Shape: (time_points, channels, n_trials)
y_data = subject['y']  # Shape: (n_trials, 1)

# Preprocess data
x_train, y_train, x_test, y_test = preprocess_data(x_data, y_data)

# Train model
# Assuming labels are 0 for left hand and 1 for right hand
l = y_train - 1
left_hand_data = x_train[l.squeeze() == 0]

left_hand_generator = train_wgan_gp(
    data=left_hand_data,
    epochs=2,
    batch_size=32,
    model_name='Left_Hand_MI'
)

r = y_train - 1
right_hand_data = x_train[r.squeeze() == 1]

right_hand_generator = Train_wgan_gp(
    data=right_hand_data,
    epochs=1,
    batch_size=32,
    model_name='Right_Hand_MI'
)

def generate_synthetic_data(generator, num_samples=100):
    noise = tf.random.normal([num_samples, 100])  # Generate noise
    synthetic_data = generator(noise, training=False)  # Generate synthetic EEG data
    return synthetic_data

# Split the processed data
# Training data is already defined as normalized_train_data and y_train
train_real_data = x_train  # Shape: (20, 22, n_timepoints)
train_real_labels = y_train - 1 # Shape: (20, 1)

# Testing data is already defined as normalized_test_data and y_test
test_real_data = x_test  # Shape: (n_test_trials, 22, n_timepoints)
test_real_labels = y_test - 1  # Shape: (n_test_trials, 1)

# Generate synthetic data
num_synthetic_samples = len(train_real_data)//2

# Generate synthetic data for each class
synthetic_left_hand_data = generate_synthetic_data(left_hand_generator, num_samples=num_synthetic_samples//2)
synthetic_right_hand_data = generate_synthetic_data(right_hand_generator, num_samples=num_synthetic_samples//2)

# Combine synthetic data
synthetic_data = np.concatenate([synthetic_left_hand_data.numpy(), synthetic_right_hand_data.numpy()], axis=0)
synthetic_labels = np.concatenate([
    np.zeros((synthetic_left_hand_data.shape[0], 1)),
    np.ones((synthetic_right_hand_data.shape[0], 1))
], axis=0)

# Combine real training data and synthetic data
augmented_train_data = np.concatenate([train_real_data, synthetic_data], axis=0)
augmented_train_labels = np.concatenate([train_real_labels.reshape(-1, 1), synthetic_labels], axis=0)

# Assuming data has shape (n_samples, channels, time_points)
train_real_data_csp = train_real_data
test_real_data_csp = test_real_data
augmented_train_data_csp = augmented_train_data

# Initialize CSP object
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

# Fit CSP on real training data and transform
csp.fit(train_real_data_csp, train_real_labels.ravel())

# Transform data
train_real_data_csp = csp.transform(train_real_data_csp)
test_real_data_csp = csp.transform(test_real_data_csp)
augmented_train_data_csp = csp.transform(augmented_train_data_csp)

# Initialize LDA classifier
lda = LinearDiscriminantAnalysis()

# Train LDA on real training data only
lda.fit(train_real_data_csp, train_real_labels.ravel())
real_data_predictions = lda.predict(test_real_data_csp)
real_data_accuracy = accuracy_score(test_real_labels, real_data_predictions)
print(f"LDA Classification Accuracy (Real Data Only): {real_data_accuracy:.4f}")

# Train LDA on augmented training data (real + synthetic)
lda.fit(augmented_train_data_csp, augmented_train_labels.ravel())
augmented_data_predictions = lda.predict(test_real_data_csp)
augmented_data_accuracy = accuracy_score(test_real_labels, augmented_data_predictions)
print(f"LDA Classification Accuracy (Augmented Data - Real + Synthetic): {augmented_data_accuracy:.4f}")

# Specify the trial and channel for comparison
trial_index = 2  # Replace with the desired trial index
channel_index = 9  # Replace with the desired channel index

# Extract data for the specific trial and channel
real_left_channel_data = left_hand_data[trial_index, channel_index, :]
synthetic_left_channel_data = synthetic_left_hand_data[trial_index, channel_index, :]

real_right_channel_data = right_hand_data[trial_index, channel_index, :]
synthetic_right_channel_data = synthetic_right_hand_data[trial_index, channel_index, :]

# Plot comparison for Left-Hand data
plt.figure(figsize=(15, 5))
plt.plot(real_left_channel_data, label='Real Left-Hand Data', linestyle='-', marker='o')
plt.plot(synthetic_left_channel_data, label='Synthetic Left-Hand Data', linestyle='--', marker='x')
plt.title(f"Left-Hand Comparison for Trial {trial_index + 1}, Channel {channel_index + 1}")
plt.xlabel("Time Points")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot comparison for Right-Hand data
plt.figure(figsize=(15, 5))
plt.plot(real_right_channel_data, label='Real Right-Hand Data', linestyle='-', marker='o')
plt.plot(synthetic_right_channel_data, label='Synthetic Right-Hand Data', linestyle='--', marker='x')
plt.title(f"Right-Hand Comparison for Trial {trial_index + 1}, Channel {channel_index + 1}")
plt.xlabel("Time Points")
plt.ylabel("Amplitude")
plt.legend()
plt.show()