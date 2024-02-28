import os
import numpy as np

def load_data(path1, path2, len_snr, N, K, SNR, data_num):
    r_label = []
    r_data = []
    for i in range(len_snr):
        file_data = os.path.join(path1, 'data_' + str(SNR[i]) + '.npy')
        file_label = os.path.join(path2, 'label_' + str(SNR[i]) + '.npy')
        r_data.append(np.load(file_data))
        r_label.append(np.load(file_label))
    data = np.reshape(r_data, (data_num, N**2))
    label = np.reshape(r_label, (data_num, K))

    return (data, label)