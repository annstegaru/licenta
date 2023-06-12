import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import scipy.stats
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA


def preprocess_data(eeg_data):
    missing_indices = np.where(np.isnan(eeg_data))[0]

# Perform linear interpolation
    for index in missing_indices:
        previous_index = index - 1
        next_index = index + 1

        # Find the previous and next non-missing values
        while np.isnan(eeg_data[previous_index]):
            previous_index -= 1
        while np.isnan(eeg_data[next_index]):
            next_index += 1

        # Perform linear interpolation
        eeg_data[index] = (eeg_data[previous_index] + eeg_data[next_index]) / 2

    return eeg_data

def extract_features(eeg_data):
    mean_features = np.mean(eeg_data, axis=1)
    std_features = np.std(eeg_data, axis=1)
    variance_features = np.var(eeg_data, axis=1)
    skewness_features = scipy.stats.skew(eeg_data, axis=1)
    kurtosis_features = scipy.stats.kurtosis(eeg_data, axis=1)

    # tsallis_entropy_features = scipy.stats.entropy(eeg_data, axis=1)
    # renyi_entropy_features = scipy.stats.entropy(eeg_data, axis=1, base=2)
    # shannon_entropy_features = scipy.stats.entropy(eeg_data, axis=1, base=np.exp(1))
    # log_energy_features = np.log(np.sum(np.square(eeg_data), axis=1))

    extracted_features = np.column_stack(
        (mean_features, std_features, variance_features, skewness_features, kurtosis_features)
        )
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

def apply_ica(data, n_components):
    ica = FastICA(n_components=n_components)
    ica.fit(data)
    components = ica.transform(data)
    return ica, components

def remove_artifacts(data, svm, ica):
    components = ica.transform(data)
    features = extract_features(components)
    artifact_labels = svm.predict(features)
    
    cleaned_data = ica.inverse_transform(components[artifact_labels == 0])
    return cleaned_data

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
    k = 6
    kf = KFold(n_splits=k)

    for train_index, val_index in kf.split(X):
        print("Training on fold:", train_index, "Validating on fold:", val_index)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.01, 0.1, 1]
        }

        # Create and train the SVM classifier
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X_train_fold, y_train_fold)
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        best_svm = SVC(**best_params)
        best_svm.fit(X_train_fold, y_train_fold)

        # Evaluate on the validation set
        val_predictions = best_svm.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, val_predictions)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print(f'Average Accuracy:{average_accuracy* 100:.3f}%')


# if __name__ == "__main__":
#     volunteers = ["1", "2", "3", "4", "5", "6"]
#     labels = ["A", "B"]
#     data = []
#     data_labels = []
    
#     for volunteer in volunteers:
#         for label in labels:
#             file_name = volunteer + "_" + label + ".mat"
#             stimuli_features = read_mat_file(file_name)
            
#             # Split the EEG data into epochs
#             epochs = split_eeg_into_epochs(stimuli_features, epoch_duration=2, overlap_ratio=0.5)
            
#             # Append the epochs to the data list
#             data.extend(epochs)
            
#             # Generate corresponding labels for each epoch
#             if label == "A":
#                 data_labels.extend([1] * len(epochs))
#             else:
#                 data_labels.extend([0] * len(epochs))
    
#     X = np.array(data)
#     y = np.array(data_labels)
#     num_epochs, epoch_length = X.shape
#     X = np.reshape(X, (num_epochs, epoch_length, 1))
#     X = np.reshape(X, (num_epochs, -1))
    
#     accuracies = []
    
#     for epoch_idx in range(len(X)):
#         print("Training on epoch:", epoch_idx)
#         X_train = np.delete(X, epoch_idx, axis=0)
#         y_train = np.delete(y, epoch_idx)
        
#         X_val = X[epoch_idx]
#         y_val = y[epoch_idx]
        
#         param_grid = {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': [0.01, 0.1, 1]
#         }

#         # Create and train the SVM classifier
#         svm = SVC()
#         grid_search = GridSearchCV(svm, param_grid, cv=5)
#         grid_search.fit(X_train, y_train)
#         best_params = grid_search.best_params_
#         print("Best Hyperparameters:", best_params)

#         best_svm = SVC(**best_params)
#         best_svm.fit(X_train, y_train)

#         # Evaluate on the validation set
#         val_predictions = best_svm.predict([X_val])
#         accuracy = accuracy_score([y_val], val_predictions)
#         accuracies.append(accuracy)
    
#     average_accuracy = np.mean(accuracies)
#     print(f'Average Accuracy: {average_accuracy * 100:.3f}%')

