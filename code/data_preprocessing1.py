import numpy as np
from scipy.signal import butter,filtfilt,decimate
from draw import *


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def testdata_preprocessing(dictc):

    data = dictc['pcg']
    inputs = [[1]]
    inputs[0]=data
    draw_test_dict={}

    inputs = np.array(data)
    # print("type(inputs)",type(inputs))
    inputs = inputs.reshape(1, -1)
    # print(inputs.shape)
    len_discarded_data = 2

    len_used_data = 2

    sampling_rate = 1000

    if (len(inputs[0]) >= sampling_rate * 5):
        # 使用了第一个心音文件的第3s的数据
        t1 = torch.tensor(inputs[0,sampling_rate * 3:sampling_rate * 4])
        # print("t1:",t1)

    else:
        t1 = None

    if inputs.shape[1] < sampling_rate * (len_discarded_data + len_used_data + 1):
        print('ERROR! The length of the data must be at least {} seconds'.format(
            len_discarded_data + len_used_data + 1))
        test_inputs = None
    else:
        path_img = r'D:/'  # 没什么用
        inv_str='123456'  # 没什么用
        draw_test_dict = drawing(dictc['pcg'], dictc['ecg'], path_img, inv_str, sampling_rate)

        inputs_pre = inputs[:,
                     sampling_rate * len_discarded_data:sampling_rate * (len_discarded_data + len_used_data)]

        lowcut = 25.0
        highcut = 400.0

        inputs = butter_bandpass_filter(inputs_pre, lowcut, highcut, sampling_rate, order=5)

        if sampling_rate == 8000:
            inputs = decimate(inputs, 8, axis=1, zero_phase=True)  # downsampling to 1000hz

        mean = np.mean(inputs, axis=1).reshape(-1, 1)
        std = np.std(inputs, axis=1).reshape(-1, 1)
        inputs = (inputs - mean) / std

        test_inputs = torch.tensor(inputs, dtype=torch.float).unsqueeze(1).unsqueeze(3)

    return test_inputs,t1,draw_test_dict



