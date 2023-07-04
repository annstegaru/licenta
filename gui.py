import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Add this line
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox

# preprocess data
def butterworth_filter(data, order, cutoff_freq, fs):
    print(data.shape)
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

def lineaer_interpolation(eeg_data):
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
    # eeg_scaled = notch_filtered_data*1.2
    # eeg_interpolated = lineaer_interpolation(notch_filtered_data)
    return notch_filtered_data

class CSVPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Signal analyser ")
        self.setGeometry(100, 100, 400, 200)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)

        self.select_file_button = QPushButton("Select File(s)")
        self.select_file_button.clicked.connect(self.select_file)

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)

        layout = QVBoxLayout()
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.plot_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def select_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        files = file_dialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")[0]
        if files:
            self.files = files
            self.file_label.setText("Selected files:\n" + "\n".join(files))

    def plot_data(self):
        if hasattr(self, "files"):
            for file in self.files:
                df = pd.read_csv(file)
                self.original_data = df.iloc[:, :22]  # Select the first 22 columns
                print(self.original_data.shape)

                num_samples = 5120  # Number of data points per column
                fs = 256.0  # Sampling frequency
                time = np.arange(num_samples) / fs  # Generate time axis

                plt.figure(figsize=(18, 12))

                for i, column in enumerate(self.original_data.columns):
                    # plt.subplot(len(self.original_data.columns), 1, i+1)
                    plt.subplot(len(self.original_data.columns), 1, i+1)
                    plt.plot(time, self.original_data[column], label="Original " + column)

                    filtered_column = preprocess_data(self.original_data)  # Apply filters to the column
                    print(filtered_column.shape)
                    plt.plot(time, filtered_column[:, i], label="Filtered " + column)
                    plt.legend()
                    plt.ylabel("Amplitude")
                    # plt.title("Column: " + column)
               
                plt.xlabel("Time")
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVPlotter()
    window.show()
    sys.exit(app.exec_())
