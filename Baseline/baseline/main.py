import os
import scipy.io
from baseline.preprocess import preprocess_data_matlab_style
from baseline.train import train_model

def main():
    # Get the path to the current script's directory
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'data.mat')  # Reference data.mat in the same directory

    # Load the .mat file
    data = scipy.io.loadmat(file_path)

    # Access the nested 'data' key
    subjects = data['data'][0]  # Array of structures for each subject

    # Initialize list to store accuracies for each subject
    accuracies = []

    for i, subject in enumerate(subjects):
        # Extract the data for the current subject
        x_data = subject['x']  # Shape: (time_points, channels, n_trials)
        labels = subject['y']  # Shape: (n_trials, 1)

        print(f"Processing Subject {i + 1}/{len(subjects)}...")

        # Preprocess data
        x_train, y_train, x_test, y_test = preprocess_data_matlab_style(x_data, labels)

        # Train and evaluate the model
        accuracy = train_model(x_train, y_train, x_test, y_test)
        accuracies.append(accuracy)

        print(f"Subject {i + 1}: LDA Classification Accuracy = {accuracy:.4f}")

    # Print all accuracies
    print("\nClassification Accuracies for All Subjects:")
    for i, acc in enumerate(accuracies):
        print(f"Subject {i + 1}: {acc:.4f}")

    # Print average accuracy
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage Classification Accuracy Across Subjects: {average_accuracy:.4f}")

if __name__ == "__main__":
    main()
