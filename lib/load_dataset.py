import os
import numpy as np
import pandas as pd
import h5py

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS07(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS07(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'NYCBike':
        data_path = os.path.join('./data/NYCBike/NYCBike.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "bike_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        # data = np.concatenate(rawdata, 1)
        data = np.stack(rawdata, -1)
    elif dataset == 'NYCTaxi':
        data_path = os.path.join('./data/NYCTaxi/NYCTaxi.h5')
        df = h5py.File(data_path, 'r')
        rawdata = []
        for feature in ["pick", "drop"]:
            key = "taxi_" + feature
            data = np.array(df[key])
            rawdata.append(data)
        data = np.stack(rawdata, -1)
    elif dataset == 'NYCTaxi':
        data_path = os.path.join('./data/NYCTaxi/NYCTaxi.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    elif dataset == '、':
        data_path = os.path.join('./data/BJTaxi/BJTaxi.npz')
        data = np.load(data_path,allow_pickle=True)['data'][:, :, :2].astype(float)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

#
# data_path = os.path.join('../data/PEMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
