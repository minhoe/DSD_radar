
import os
import numpy as np
import scipy.io
import math

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataLoader:
    def __init__(self, data_path, data_file, mode, exp, log, X_scaler=None, Y_scaler=None):

        assert data_file.endswith('mat')
        assert mode in ['trn', 'val', 'tst']

        self.data_path = data_path
        self.trn_file = data_file
        self.mode = mode
        self.exp = exp
        self.log = log

        if X_scaler:
            self.X_scaler = X_scaler
        else:
            self.X_scaler = None

        if Y_scaler:
            self.Y_scaler = Y_scaler
        else:
            self.Y_scaler = None

    def get_data(self):
        # Load data
        data_x_pd, data_y_pd = self._input_loader()
        return data_x_pd, data_y_pd

    def _input_loader(self):
        # Load *.mat dataset
        data = scipy.io.loadmat(os.path.join(self.data_path, self.trn_file))

        # Divide into the training set and validation set
        if self.exp == 'EXP1':
            ##### CASE 1. 2 x 2 ----------------------------------------------------------------------------------------
            if self.mode == 'trn':
                data_x1_pd, data_x2_pd = data['trn_refl'], data['trn_dif']
                data_y1_pd, data_y2_pd = data['trn_dm'],   data['trn_w']
            elif self.mode == 'val':
                data_x1_pd, data_x2_pd = data['val_refl'], data['val_dif']
                data_y1_pd, data_y2_pd = data['val_dm'],   data['val_w']
            else:
                data_x1_pd, data_x2_pd = data['test_refl'], data['test_dif']
                data_y1_pd, data_y2_pd = data['test_dm'],   data['test_w']

            # Logarithm of values
            if self.log:
                data_x1_pd, data_x2_pd = np.log(data_x1_pd + 0.001), np.log(data_x2_pd + 0.001)
                data_y1_pd, data_y2_pd = np.log(data_y1_pd + 0.001), np.log(data_y2_pd + 0.001)

            # Concatenate & Transpose variables
            data_x_pd = np.transpose(np.concatenate([data_x1_pd, data_x2_pd]))
            data_y_pd = np.transpose(np.concatenate([data_y1_pd, data_y2_pd]))

        elif self.exp == 'EXP2':
            ##### CASE 2. 2 x 1 --------------------------------------------------------------------------------
            if self.mode == 'trn':
                data_x1_pd, data_x2_pd = data['trn_refl'], data['trn_dif']
                data_y1_pd = data['trn_r']
            elif self.mode == 'val':
                data_x1_pd, data_x2_pd = data['val_refl'], data['val_dif']
                data_y1_pd = data['val_r']
            else:
                data_x1_pd, data_x2_pd = data['test_refl'], data['test_dif']
                data_y1_pd = data['test_r']

            # Logarithm of values
            if self.log:
                data_x1_pd, data_x2_pd = np.log(data_x1_pd + 0.001), np.log(data_x2_pd + 0.001)
                data_y1_pd = np.log(data_y1_pd + 0.001)

            # Concatenate & Transpose variables
            data_x_pd = np.transpose(np.concatenate([data_x1_pd, data_x2_pd]))
            data_y_pd = np.transpose(data_y1_pd)

        elif self.exp == 'EXP3':
            ##### CASE 3. 3 x 1 --------------------------------------------------------------------------------
            if self.mode == 'trn':
                data_x1_pd, data_x2_pd, data_x3_pd = data['trn_refl'], data['trn_dif'], data['trn_cc']
                data_y1_pd = data['trn_r']
            elif self.mode == 'val':
                data_x1_pd, data_x2_pd, data_x3_pd = data['val_refl'], data['val_dif'], data['val_cc']
                data_y1_pd = data['val_r']
            else:
                data_x1_pd, data_x2_pd, data_x3_pd = data['test_refl'], data['test_dif'], data['test_cc']
                data_y1_pd = data['test_r']

            # Logarithm of values
            if self.log:
                data_x1_pd, data_x2_pd, data_x3pd = np.log(data_x1_pd + 0.001), np.log(data_x2_pd + 0.001), np.log(data_x3_pd + 0.001)
                data_y1_pd = np.log(data_y1_pd + 0.001)

            # Concatenate & Transpose variables
            data_x_pd = np.transpose(np.concatenate([data_x1_pd, data_x2_pd, data_x3_pd]))
            data_y_pd = np.transpose(data_y1_pd)

        else:
            pass

        print(self.mode, data_x_pd.min(axis=0), data_x_pd.max(axis=0), data_x_pd.mean(axis=0), data_x_pd.std(axis=0))

        # Scaling
        if self.X_scaler is None:
            scaler = StandardScaler()
            self.X_scaler = scaler.fit(data_x_pd)
            #self.Y_scaler = scaler.fit(data_y_pd)

        data_x_pd = self.X_scaler.transform(data_x_pd)
        #data_y_pd = self.Y_scaler.transform(data_y_pd)

        return data_x_pd, data_y_pd


class DSD_radar_dataset(Dataset):
    def __init__(self, data_path, data_file, mode, exp, log, X_scaler=None, Y_scaler=None):
        # Load Data
        DL = DataLoader(data_path, data_file, mode, exp, log, X_scaler, Y_scaler)
        self.x, self.y, = DL.get_data()

        print('Loading data : X : ', self.x.shape, 'Y :', self.y.shape)

        self.X_scaler = DL.X_scaler
        self.Y_scaler = DL.Y_scaler

    def get_x_shape(self):
        return self.x.shape

    def get_y_shape(self):
        return self.y.shape

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        return x, y

    def __len__(self):
        return len(self.x)