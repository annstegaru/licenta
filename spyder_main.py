# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:33:37 2023

@author: anne_
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import mne
import pandas as pd


fname = "D:/Anne/Facultate/licenta/ScienceDirect_files_16Dec2022_10-38-19/CSV/sig1Hz_V1_A.csv"

landmarks_frame = pd.read_csv(fname)
