import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Step 1: Preprocessing
def preprocess_data(eeg_data):

    #standardization
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(eeg_data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(preprocessed_data)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Preprocessed EEG Data')
    # plt.show()
    return preprocessed_data

# Step 2: Feature Extraction
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
    
    extracted_features = np.column_stack((mean_features, std_features, variance_features, skewness_features, kurtosis_features))
    return extracted_features

# Step 3: Data Split
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

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
    print("Preprocessed data format:")
    print(np.array(data).shape)
    print(data_labels)
    print("")

    X = np.array(data)
    y = np.array(data_labels)
    X = np.reshape(X, (X.shape[0], X.shape[1], -1))

    input_size = X.shape[2]
    hidden_size = 64
    output_size = 1

    model = LSTMModel(input_size, hidden_size, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())


    accuracies = []
    k = 5
    kf = KFold(n_splits=k)

    for train_index, val_index in kf.split(X):
        print("Training on fold:", train_index, "Validating on fold:", val_index)
        X_train_fold, X_val_fold = torch.from_numpy(X[train_index]), torch.from_numpy(X[val_index])
        y_train_fold, y_val_fold = torch.from_numpy(y[train_index]), torch.from_numpy(y[val_index])

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_fold.float())
        loss = criterion(outputs.squeeze(), y_train_fold.float())
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold.float())
            val_predictions = (val_outputs.squeeze() > 0.5).float()
            accuracy = accuracy_score(y_val_fold, val_predictions.numpy())
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print("Average Accuracy:", average_accuracy*100)
