import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.stats
from sklearn.model_selection import KFold

def preprocess_data(eeg_data):
    # Standardization
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(eeg_data)
    return preprocessed_data

def extract_features(eeg_data):
    mean_features = np.mean(eeg_data, axis=1)
    std_features = np.std(eeg_data, axis=1)
    variance_features = np.var(eeg_data, axis=1)
    skewness_features = scipy.stats.skew(eeg_data, axis=1)
    kurtosis_features = scipy.stats.kurtosis(eeg_data, axis=1)

    extracted_features = np.column_stack((mean_features, std_features, variance_features, skewness_features, kurtosis_features))
    return extracted_features

def split_data(features, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
    return X_train, X_test, y_train, y_test

def read_mat_file(file_name):
    mat = sio.loadmat(file_name)
    mat = mat["ssvep"]
    data = []
    for stimuli in mat:
        stimuli_name = stimuli[1][0]
        stimuli_data = np.array(stimuli[0])
        stimuli_data = np.delete(stimuli_data, -1, axis=1)
        stimuli_data = preprocess_data(stimuli_data)
        stimuli_features = extract_features(stimuli_data)
        data.append(stimuli_features)
    return data

if __name__ == "__main__":
    volunteers = ["1", "2", "3", "4", "5", "6"]
    labels = ["A", "B"]
    data = []
    data_labels = []
    for volunteer in volunteers:
        for label in labels:
            file_name = volunteer + "_" + label + ".mat"
            stimuli_features = read_mat_file(file_name)
            data.append(stimuli_features)
            if label == "A":
                data_labels.append(1)
            else:
                data_labels.append(0)

    X = np.array(data)
    y = np.array(data_labels)
    X = np.reshape(X, (X.shape[0], X.shape[1], -1))
    X = np.reshape(X, (X.shape[0], -1))  # Flatten the input data

    accuracies = []
    epochs = 5  # Number of epochs for training
    k = 5
    kf = KFold(n_splits=k)

    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        for train_index, val_index in kf.split(X):
            print("Training on fold:", train_index, "Validating on fold:", val_index)
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Create and train the SVM classifier
            svm = SVC()
            svm.fit(X_train_fold, y_train_fold)

            # Evaluate on the validation set
            val_predictions = svm.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, val_predictions)
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print("Average Accuracy:", average_accuracy * 100)
