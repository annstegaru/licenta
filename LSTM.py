import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy import signal
import pywt


def butterworth_filter(data, order, cutoff_freq, fs):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

def notch_filter(data, notch_freq, q_factor, fs):
    b, a = iirnotch(notch_freq, q_factor, fs=fs)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

def high_pass_filter(data, order, cutoff_freq, sampling_rate):
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff_freq, btype='high', analog=False, output='ba')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def preprocess_data(eeg_data):
    fs = 256  # Sampling frequency
    order = 6  # Filter order
    cutoff_freq_low = 40  # Cutoff frequency for low pass
    cutoff_freq_high = 0.5  # Cutoff frequency for high pass
    notch_freq = 50  # Notch frequency
    q_factor = 30  # Quality factor
    filtered_data = butterworth_filter(eeg_data, order, cutoff_freq_low, fs)
    filtered_data_high = high_pass_filter(filtered_data, order, cutoff_freq_high, fs)
    notch_filtered_data = notch_filter(filtered_data_high, notch_freq, q_factor, fs)
    eeg_scaled = notch_filtered_data*1.2
    return eeg_scaled

def mean(x):
    return np.array(np.mean(x))

def std(x):
    return np.array(np.std(x))

def ptp(x):
    return np.array(np.ptp(x))

def var(x):
    return np.array(np.var(x))

def minim(x):
    return np.array(np.min(x))

def maxim(x):
    return np.array(np.max(x))

def argminim(x):
    return np.array(np.argmin(x))

def argmaxim(x):
    return np.array(np.argmax(x))

def abs_diff_signal(x):
    return np.array(np.sum(np.abs(np.diff(x)),axis=-1))

def skewness(x):
    return np.array(stats.skew(x))

def kurtosis(x):
    return np.array(stats.kurtosis(x,axis=-1))

def signal_energy(signal):
    signal = np.asarray(signal)
    energy = np.sum(np.square(signal))
    return energy

def spectral_entropy(x):
    f, Pxx = signal.welch(x, fs=256, nperseg=128, noverlap=None)
    normalized_Pxx = Pxx / np.sum(Pxx)  # Normalize the power spectrum
    entropy = -np.sum(normalized_Pxx * np.log2(normalized_Pxx))
    return entropy

def wavelet_features(signal):
    wavelet='db4'
    levels=4
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    features = np.hstack(coeffs)
    return features

def concatenate_features(x):
    return np.hstack((mean(x), std(x), ptp(x), var(x), minim(x), maxim(x), argminim(x), 
                           argmaxim(x), abs_diff_signal(x), skewness(x), kurtosis(x), signal_energy(x), spectral_entropy(x), wavelet_features(x)))

def extract_features(x):
    # receives a (1024, 22) matrix which represents 22 channels of 1024 samples each
    # returns a (1, 264) matrix which represents 22 channels of 12 features each
    # features: mean, std, ptp, var, min, max, argmin, argmax, rms, abs_diff_signal, skewness, kurtosis
    x = np.transpose(x)
    output = []
    for channel in x:
        # print(concatenate_features(channel).shape)
        output.append(concatenate_features(channel))
    return np.array(np.hstack(output))

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
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional LSTM
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size).to(inputs.device)  # Multiply by 2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size).to(inputs.device)

        # Forward pass through LSTM layers
        lstm_out, _ = self.lstm(inputs, (h0, c0))

        # Extract the last time step output from the LSTM layers
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
    num_layers = 2  # Specify the number of LSTM layers
    dropout = 0.2  # Set the dropout rate
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    accuracies = []
    k = 6
    kf = KFold(n_splits=k)

    for train_index, val_index in kf.split(X):
        print("Training on fold:", train_index, "Validating on fold:", val_index)
        X_train_fold, X_val_fold = torch.from_numpy(X[train_index]), torch.from_numpy(X[val_index])
        y_train_fold, y_val_fold = torch.from_numpy(y[train_index]), torch.from_numpy(y[val_index])
        y_train_fold = y_train_fold.view(-1, 1)  # Reshape target tensor for binary classification

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
            y_val_fold = y_val_fold.view(-1, 1)  # Reshape target tensor for binary classification
            accuracy = accuracy_score(y_val_fold, val_predictions.numpy())
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print("Average Accuracy:", average_accuracy*100)
