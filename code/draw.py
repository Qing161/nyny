from decimal import Decimal
import shutil

import pywt

from scipy.signal import hilbert, chirp, argrelextrema, cheby1
from scipy.fftpack import fft, fftshift, ifft
import warnings

warnings.filterwarnings("ignore")
from scipy import signal

import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd

import torch

import random

import warnings

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.backends.cudnn.benchmark = True

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# FFT变换代码---------------------------------------------------------------------------------------------------------------------------------------------
def apply_fft_pw(data, smaple_freq):
    N = len(data)

    fft_data = fft(data)

    fft_amp0 = np.array(np.abs(fft_data) / N * 2)

    fft_amp0[0] = 0 * fft_amp0[0]

    fft_amp1 = fft_amp0[0:int(N / 2)]

    list1 = np.array(range(0, int(N / 2)))

    freq1 = smaple_freq * list1 / N

    f_base = freq1[np.argmax(fft_amp1)]

    return f_base, freq1, fft_amp1


# 包络代码------------------------------------------------------------------------------------------------------------------------------------------------
def Envelop(data, num):
    for i in range(num):
        analytic_signal = hilbert(data)
        data = np.abs(analytic_signal)
    return data


# 波形绘制代码------------------------------------------------------------------------------------------------------------------------------------------
def plot_y(y, title=None):
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(y, linewidth=1)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()


def plot_xy(x, y, title=None):
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(x, y, linewidth=1)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()


# 滤波代码-----------------------------------------------------------------------------------------------------------------------------------------------
# 心电滤波代码1
def denoise_ecg(data, wavelet="sym8", level=10):
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    lis_denoise = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


def denoise_ecg2(data, wavelet="sym8", level=10):
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    lis_denoise = [0]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


def denoise_pcg(data, wavelet="sym8", level=10):
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    lis_denoise = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 滤波代码-----------------------------------------------------------------------------------------------------------------------------------------------
# 心音滤波代码2
def denoise_pcg2(data, wavelet="sym8", level=10):
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    cA10.fill(0)
    cD10.fill(0)
    cD1.fill(0)
    cD2.fill(0)
    cD3.fill(0)
    lis_denoise = [2, 3, 4, 5, 6, 7]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


def denoise_pcg3(data, wavelet="sym10", level=10):
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    lis_denoise = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 识别算法-----------------------------------------------------------------------------------------------------------------------------------------------
# 心音识别S1，S2
def pcg_s1_s2(data, sample_freq, f_base):
    NT_base = int((1 / f_base) * sample_freq)
    p1 = []
    p2 = []
    temp = np.argmax(data[0:int(NT_base * 1)])
    # 注意这里的 NT_base*1.34 是为了防止 temp1+int(NT_base/3) 访问出界，因为有可能temp1就是靠右的边界值
    while temp + int(NT_base * 1.5) < len(data):
        temp1 = np.argmax(data[temp + int(NT_base / 2):temp + int(NT_base * 1)]) + temp + int(NT_base / 2)
        temp2_1 = np.argmax(data[temp1 - int(NT_base / 2.5):temp1 - int(NT_base / 8)]) + temp1 - int(NT_base / 2.5)
        temp2_2 = np.argmax(data[temp1 + int(NT_base / 8):temp1 + int(NT_base / 2.5)]) + temp1 + int(NT_base / 8)
        if data[temp2_1] > data[temp2_2]:
            temp2 = temp1
            temp1 = temp2_1
        else:
            temp2 = temp2_2
        p1.append(temp1)
        p2.append(temp2)
        temp = p2[-1]
    return p1, p2


# 心电识别R波
def ecg_r(data, sample_freq, f_base):
    f_base = f_base
    NT_base = int((1 / f_base) * sample_freq)
    r = [np.argmax(data[0:int(NT_base * 1.1)])]
    while r[-1] + int(NT_base * 1.1) < len(data):
        temp = np.argmax(data[r[-1] + int(NT_base / 2):r[-1] + int(NT_base * 1.2)]) + r[-1] + int(NT_base / 2)
        r.append(temp)
    return r[1:-1]


# 寻找s1,s2波谷点
def find_s1_s2_lr(data_o, sample_freq):
    S1_L = []
    S1_R = []
    S2_L = []
    S2_R = []
    # S1,S2
    pcg = Envelop(denoise_pcg(data_o), 3)
    f_base, freq1, fft_amp1 = apply_fft_pw(pcg, sample_freq)
    if f_base < 0.5 or f_base > 1.5:
        f_base = 1
    N_base = int((1 / f_base) * sample_freq)
    s1, s2 = pcg_s1_s2(data_o, sample_freq, f_base)
    # 波谷
    #     data_o3 = data_o
    data_o3 = Envelop(denoise_pcg2(data_o), 1)
    for i in range(len(s1)):
        temp = 0.08
        S1_L.append(
            np.argmin(data_o3[s1[i] - int(N_base * temp):s1[i] - int(N_base * temp / 4)]) + s1[i] - int(N_base * temp))
        S1_R.append(np.argmin(data_o3[s1[i] + int(N_base * temp / 4):s1[i] + int(N_base * temp)]) + s1[i] + int(
            N_base * temp / 4))
        S2_L.append(
            np.argmin(data_o3[s2[i] - int(N_base * temp):s2[i] - int(N_base * temp / 4)]) + s2[i] - int(N_base * temp))
        S2_R.append(np.argmin(data_o3[s2[i] + int(N_base * temp / 4):s2[i] + int(N_base * temp)]) + s2[i] + int(
            N_base * temp / 4))
    return f_base, s1, s2, S1_L, S1_R, S2_L, S2_R


def cal_p_f(data_o, S1_L, S1_R, S2_L, S2_R, sample_freq):
    data_test = denoise_pcg3(data_o)
    f = []
    p_f = []
    for i in range(len(S1_L) - 1):
        fft_0 = fft(data_test[S1_L[i]:S1_L[i + 1]], sample_freq)
        f0 = np.mean(np.power(np.abs(fft_0[1:int(sample_freq / 2)]), 2))
        fft_1 = fft(data_test[S1_L[i]:S1_R[i]], sample_freq)
        f1 = np.mean(np.power(np.abs(fft_1[1:int(sample_freq / 2)]), 2))
        fft_2 = fft(data_test[S2_L[i]:S2_R[i]], sample_freq)
        f2 = np.mean(np.power(np.abs(fft_2[1:int(sample_freq / 2)]), 2))
        fft_3 = fft(data_test[S1_R[i]:S2_L[i]], sample_freq)
        f3 = np.mean(np.power(np.abs(fft_3[1:int(sample_freq / 2)]), 2))
        fft_4 = fft(data_test[S2_R[i]:S1_L[i + 1]], sample_freq)
        f4 = np.mean(np.power(np.abs(fft_4[1:int(sample_freq / 2)]), 2))
        f.append([f0, f1 + f2 + f3 + f4, f1, f2, f3, f4])
    for i in range(4):
        p_f.append(np.mean((np.array(f).T[i + 2]) / (np.array(f).T[1])))
    return p_f


def find_r_pt_lr(sample_freq, data_o):
    P_peak = []
    PL_valley = []
    PR_valley = []
    T_peak = []
    TL_valley = []
    TR_valley = []

    ecg = Envelop(denoise_ecg(data_o), 3)
    f_base, freq1, fft_amp1 = apply_fft_pw(ecg, 1000)

    if f_base < 0.5 or f_base > 1.5:
        f_base = 1
    N_base = int((1 / f_base) * sample_freq)

    R_peak = ecg_r(data_o, sample_freq, f_base)

    data_o2 = denoise_ecg2(data_o)
    for i in range(len(R_peak)):
        temp1 = np.argmax(data_o2[R_peak[i] - int(N_base * 0.2):R_peak[i] - int(N_base * 0.05)]) + R_peak[i] - int(
            N_base * 0.2)
        P_peak.append(temp1)
        temp2 = np.argmin(data_o2[P_peak[-1] - int(N_base * 0.1):P_peak[-1]]) + P_peak[-1] - int(N_base * 0.1)
        PL_valley.append(temp2)
        temp3 = np.argmin(data_o2[P_peak[-1]:P_peak[-1] + int(N_base * 0.1)]) + P_peak[-1]
        PR_valley.append(temp3)

        temp1 = np.argmax(data_o2[R_peak[i] + int(N_base * 0.15):R_peak[i] + int(N_base * 0.4)]) + R_peak[i] + int(
            N_base * 0.15)
        T_peak.append(temp1)
        temp2 = np.argmin(data_o2[T_peak[-1] - int(N_base * 0.2):T_peak[-1]]) + T_peak[-1] - int(N_base * 0.2)
        TL_valley.append(temp2)
        temp3 = np.argmin(data_o2[T_peak[-1]:T_peak[-1] + int(N_base * 0.2)]) + T_peak[-1]
        TR_valley.append(temp3)
    return f_base, R_peak, P_peak, PL_valley, PR_valley, T_peak, TL_valley, TR_valley


# 波形分解查看代码-----------------------------------------------------------------------------------------------------------------------------------------
def see_resolve(data, wavelet="sym8", level=5):
    """
    这个函数本质上是我用来做测试的，在实际使用中不用，服务器上更用不到。
    """
    # 确保只显示图形
    InteractiveShell.ast_node_interactivity = "last_expr"
    # 分解
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和九个高频的细节分量
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    # cA8,cD8,cA7,cD7,cA6,cD6,cA5,cD5, cA4,cD4, cA3,cD3,cA2,cD2,cA1,cD1 = coeffs
    fig1 = plt.figure(figsize=(15, 15))

    ax1 = fig1.add_subplot(431)
    ax2 = fig1.add_subplot(432)
    ax3 = fig1.add_subplot(433)
    ax4 = fig1.add_subplot(434)
    ax5 = fig1.add_subplot(435)
    ax6 = fig1.add_subplot(436)
    # ax7 = fig1.add_subplot(437)
    # ax8 = fig1.add_subplot(438)
    # ax9 = fig1.add_subplot(439)
    # ax10 = fig1.add_subplot(4,3,10)

    ax1.set_title("cA5")
    ax1.plot(cA5)
    ax2.set_title("cD1")
    ax2.plot(cD1)
    ax3.set_title("cD2")
    ax3.plot(cD2)
    ax4.set_title("cD3")
    ax4.plot(cD3)
    ax5.set_title("cD4")
    ax5.plot(cD4)
    ax6.set_title("cD5")
    ax6.plot(cD5)
    # ax7.set_title("cD6")
    # ax7.plot(cD6)
    # ax8.set_title("cD7")
    # ax8.plot(cD7)
    # ax9.set_title("cD8")
    # ax9.plot(cD8)


# FFT变换代码---------------------------------------------------------------------------------------------------------------------------------------------
def apply_fft_pw(data, smaple_freq):
    """
    这是对FFT代码的变体，确保得到的结果符合我们的预期。
    """
    N = len(data)
    # 进行FFT变换
    fft_data = fft(data)
    # 取绝对值，并归一化
    fft_amp0 = np.array(np.abs(fft_data) / N * 2)
    # 0Hz频率 FFT变换完第一个数时0Hz频率,0Hz就是没有波动,没有波动有个专业一点的说法,叫直流分量
    # 乘以0以防值过大，对后续判断造成影响（之前乘以了0.5发现会影响我们的结果）
    fft_amp0[0] = 0 * fft_amp0[0]
    # 只需要一半的数据，因为是FFT变换完是对称的
    fft_amp1 = fft_amp0[0:int(N / 2)]
    # 构造绘图所用的x
    list1 = np.array(range(0, int(N / 2)))
    # 转换成频率，注意精度为 0-sample_fre/2 (hz)  例如原始采样率为1000hz，经过fft变换完之后，最大频率为500hz
    freq1 = smaple_freq * list1 / N
    # 对最大值点查看对应的频率为多少  最大值点和最大的频率点是不一样的，那个频率点的值大就说明在改点的能量较高
    f_base = freq1[np.argmax(fft_amp1)]
    # 返回了三个参数，最大值点的频率，频率点列表以及fft变换之后的一半数据
    return f_base, freq1, fft_amp1


# 包络代码------------------------------------------------------------------------------------------------------------------------------------------------
def Envelop(data, num):
    """
    data:待绘制包络的数据
    num: 重复的次数
    注意：需要传入去除基线漂移的信号，也要去除直流分量，也就是保证信号的基线在零,而对于心音信号来说，则需要对分解出来的所有系数进行滤波
    """
    for i in range(num):
        # 传入数据并使用hilbert变换
        analytic_signal = hilbert(data)
        # 进行绝对值化，对于心音来说其实就是把在x轴下方的数据搬了上来，和上面的趋势进行重叠。
        data = np.abs(analytic_signal)
    return data


# 波形绘制代码------------------------------------------------------------------------------------------------------------------------------------------
def plot_y(y, title=None):
    # 只需要传入y轴数据即可
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(y, linewidth=1)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()


# plot_signal(amplitude_envelope, title='envelope')

def plot_xy(x, y, title=None):
    # 需要传入x轴，y轴数据
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(x, y, linewidth=1)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()


# plot_signal(amplitude_envelope, title='envelope')


# 滤波代码-----------------------------------------------------------------------------------------------------------------------------------------------
# 心电滤波代码
def denoise_ecg(data, wavelet="sym8", level=10):
    """
    这个滤除心电噪音的代码是对所有分量进行了滤波，较为极致，具体可以看下面效果
    """
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和10个高频的细节分量：
    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 0      1    2    3    4    5    6    7    8    9   10
    lis_denoise = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 心电滤波代码2
def denoise_ecg2(data, wavelet="sym8", level=10):
    """
    这个滤除心电噪音的代码只对一个低频分量进行了滤波
    """
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和八个高频的细节分量：
    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 0      1    2    3    4    5    6    7    8    9   10
    lis_denoise = [0]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 心音滤波代码
def denoise_pcg(data, wavelet="sym8", level=10):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和九个高频的细节分量：
    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 0      1    2    3    4    5    6    7    8    9   10
    lis_denoise = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 滤波代码-----------------------------------------------------------------------------------------------------------------------------------------------
# 心音滤波代码2
def denoise_pcg2(data, wavelet="sym8", level=10):
    # 将数据变为 2^8的倍数才可以计算
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和八个高频的细节分量：
    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 0    1    2   3   4   5   6   7   8   9   10
    cA10.fill(0)
    cD10.fill(0)
    cD1.fill(0)
    cD2.fill(0)
    cD3.fill(0)
    lis_denoise = [2, 3, 4, 5, 6, 7]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


def denoise_pcg3(data, wavelet="sym10", level=10):
    # 将数据变为 2^8的倍数才可以计算
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    # 上面的代码将原信号分解为一个低频的近似分量和八个高频的细节分量：
    cA10, cD10, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 0    1    2   3   4   5   6   7   8   9   10
    lis_denoise = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    for i in lis_denoise:
        threshold = (np.median(np.abs(coeffs[i])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[i]))))
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return rdata


# 滤波器-----------------------------------------------------------------------------------------------------------------------------------
# 切比雪夫滤波
def apply_chebyshev_filter(data, fs, ftype, freqs=[], order=4, rp=0.5):
    nyq = 0.5 * fs
    if ftype == 'low_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.cheby1(order, rp, cut, btype='lowpass')
    elif ftype == 'high_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.cheby1(order, rp, cut, btype='highpass')
    elif ftype == 'band_pass':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.cheby1(order, rp, [lowcut, highcut], btype='bandpass')
    elif ftype == 'band_stop':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.cheby1(order, rp, [lowcut, highcut], btype='bandstop')

    filtered = signal.filtfilt(b, a, data)
    return filtered


# 巴特沃斯滤波
def apply_butter_filter(data, fs, ftype, freqs=[], order=4):
    nyq = 0.5 * fs
    if ftype == 'low_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.butter(order, cut, btype='lowpass')
    elif ftype == 'high_pass':
        assert len(freqs) == 1
        cut = freqs[0] / nyq
        b, a = signal.butter(order, cut, btype='highpass')
    elif ftype == 'band_pass':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.butter(order, [lowcut, highcut], btype='bandpass')
    elif ftype == 'band_stop':
        assert len(freqs) == 2
        lowcut, highcut = freqs[0] / nyq, freqs[1] / nyq
        b, a = signal.butter(order, [lowcut, highcut], btype='bandstop')

    filtered = signal.filtfilt(b, a, data)
    return filtered


# 对台式心电数据处理并保存
def read_save_ecg12(path_read, path_save):
    """
    Time CH1 CH2 CH3 CH4 CH5 CH6 CH7 CH8
    1路（I）：CH2
    2路（II）：CH3
    3路（III）：CH3-CH2
    4路（aVR）：-0.5* (CH2+CH3)
    5路（aVL）：CH2-0.5* CH3
    6路（aVF）：CH3-0.5* CH2
    7路（V1）：CH8
    8路（V2）：CH4
    9路（V3）：CH5
    10路（V4）：CH6
    11路（V5）：CH7
    12路（V6）：CH1

    处理完成后的数组第一列是时间，其余为1-12路心电
    """
    data_ecg = pd.read_csv(path_read, header=None)
    data_ecg = data_ecg.loc[1000:]  # 删除掉前面1s的数据
    # 这里的减一是减去第一秒的数据后，时间需要向前推进1秒
    data_ecg12 = []
    data_ecg12.append(data_ecg[0] - 1)
    data_ecg12.append(data_ecg[2])  # 这儿开始是第一路
    data_ecg12.append(data_ecg[3])
    data_ecg12.append(data_ecg[3] - data_ecg[2])
    data_ecg12.append(-0.5 * (data_ecg[2] + data_ecg[3]))
    data_ecg12.append(data_ecg[2] - 0.5 * data_ecg[3])
    data_ecg12.append(data_ecg[3] - 0.5 * data_ecg[2])
    data_ecg12.append(data_ecg[8])
    data_ecg12.append(data_ecg[4])
    data_ecg12.append(data_ecg[5])
    data_ecg12.append(data_ecg[6])
    data_ecg12.append(data_ecg[7])
    data_ecg12.append(data_ecg[1])
    # 确保转换完的数据尺寸和原数据相对应
    assert data_ecg.shape[0] == np.array(data_ecg12).T.shape[0]
    data = np.array(data_ecg12).T
    # 保存csv文件到对应文件夹
    DF = pd.DataFrame(data)
    DF.to_csv(path_save, index=False)
    return DF


# path_read = r"C:\Users\Administrator.DESKTOP-P17BHR8\Desktop\ecg3.csv"
# path_save = r"C:\Users\Administrator.DESKTOP-P17BHR8\Desktop\ecg3s.csv"
# DF = read_save_ecg12(path_read, path_save)

# 识别算法-----------------------------------------------------------------------------------------------------------------------------------------------
# 心音识别S1，S2
def pcg_s1_s2(data, sample_freq, f_base):
    """
    在做S1,S2识别过程中其实试验了很多方法，网络也查了很多，都没有一个较为理想的方案，后续的试验师弟可以考虑从论文中用的大多数数学方法进行，当然论文所用的一些方法复现难度会很大，且大概率会和作者所得到的结果不一致，另外论文中所用的方法打都适用于他们的数据集，而咋们的不一定满足他们的数据要求。
    刚开始做S1,S2识别时，想到的方法为阈值隔断，这和廉老师他们假期开会学生提到的方法是一样的，显然单纯用阈值隔断可能能识别到所有的S1,但是对S2的识别就难以捉摸了。所以在测试了多组数据后，我放弃了这个方案。
    接着我开始尝试进行加窗阈值隔断，也就是先识别一个窗口内的S1，S2，在去下一个窗口，但是这个窗口单纯自己定义实在是过于浮夸，且对于不同采样率的数据，没有很强的适应性，在尝试了多次后也放弃了。
    后来经过和一个师弟的讨论后，采用了频域上的加窗方法，具体为下面的算法。


    算法开始：
    注意到这个算法里面的数据务必保证心音至少包含三个周期。
    data_o：需要识别的心音数据
    sample_freq：心音数据的采样率
    f_base：做完FFT变换之后所使用的最大频率值。这个值需要重点注意一下，通常情况下我们对心音信号进行FFT变换得到的频率是在80-90hz之间，因为心音就是在这个频段能量最高，但是我们现在想要的是心音的周期性频率，也就是1s钟会有几个心跳，一次心跳代表的就是一个S1和一个S2，所以是需要先对心音信号做包络（在这个函数中我们就可以看到find_s1_s2_lr(data_o,sample_freq)），因为做完包络就只剩下了大的趋势，细节信息已经没有了，此时很容易就获得了心音的周期频率。经过对多组数据测试证明，大数据情况我们都可以获得准确的心音周期，但是也有一些质量很差的信号，算法会计算错误，此时就需要矫正了，这个我在find_s1_s2_lr(data_o,sample_freq)函数中也做了，具体可以看看。

    首先需要考虑s1,s2的高低问题，因为我们在采集过程中s1会出现忽而低，忽而高的情况
    我们这里考虑的时候，先去除掉第一个心音周期
    循环开始：紧接着我们先使用窗口找寻相邻周期的最大值，无论是s1，还是s2都无所谓
              继而在最大值周围附近搜索第二个极大值
              判断两个极大值的位置关系，小的为s1，大的为s2
              放入数组，并将s2的位置对应的x值赋值给temp
              进行下一轮循环
    """
    # (1/f_base 即为一个周期的时间，例如f_base为2hz，那么一个周期的时间就是0.5s，接着乘以采样率就是一个完整周期包含的采样点个数了。
    NT_base = int((1 / f_base) * sample_freq)
    # p1数组用来存储S1点对应的位置
    p1 = []
    # p2数组用来存S2
    p2 = []
    # 寻找初始窗口内的最大值，肯定是S1，
    temp = np.argmax(data[0:int(NT_base * 1)])
    # 注意这里的 NT_base*1.5 是为了防止出界，也就是我们不使用最后一个周期，因为大概率最后一个周期是不完整的。
    while temp + int(NT_base * 1.5) < len(data):
        #  + temp + int(NT_base/2) 加这两个值是进行位置的变化，  np.argmax(data[temp+int(NT_base/2):temp+int(NT_base*1)]) 这是在寻找temp旁边的一个小窗口中的最大值，无法确实他是s1还是s2，因为有可能窗口没有达到下一个S1，那么此时temp1就是S2了，如果窗口到达了S1，那temp1就是S1了。
        # 那既然无法确定，我们就需要自己确定一下了，具体方法当然是我们在temp1的左边进行寻找，在temp1的右边进行寻找，因为收缩期时间较小于舒张期时间，所以当temp1是S2时，我们在左边寻找到的值肯定是S1，而在右边寻找到的值会小很多，当temp1是S1时，我们在右边寻找到的是S2，在左边寻找到的值会小得多。

        # 注意：只有在第一次寻找时无法确定temp1是S1还是S2，当进行完一次循环后，temp就更新为最新的S2位置了，那每次找到的temp1就一定是S1。
        # 另外这套算法其实有点过于繁琐了，改进的思路有很多，例如将temp1 = np.argmax(data[temp+int(NT_base/2):temp+int(NT_base*1.2)]) + temp + int(NT_base/2),此时就能确保temp1一定是S1了。师弟后续有能力自己进行改进，提高代码效率。
        temp1 = np.argmax(data[temp + int(NT_base / 2):temp + int(NT_base * 1)]) + temp + int(NT_base / 2)
        # 在temp1的左边寻找极大值点，也是在一个小窗口内，注意这个点可能不是s1或s2
        temp2_1 = np.argmax(data[temp1 - int(NT_base / 2.5):temp1 - int(NT_base / 8)]) + temp1 - int(NT_base / 2.5)
        # 在temp1的右边寻找极大值点，也是在一个小窗口内吗，注意这个点可能不是s1或s2
        temp2_2 = np.argmax(data[temp1 + int(NT_base / 8):temp1 + int(NT_base / 2.5)]) + temp1 + int(NT_base / 8)
        if data[temp2_1] > data[temp2_2]:
            temp2 = temp1
            temp1 = temp2_1
        else:
            temp2 = temp2_2
        p1.append(temp1)
        p2.append(temp2)
        # temp的值进行更新，现在我们就能确保temp每次都是S2了。
        temp = p2[-1]
    return p1, p2


# 心电识别R波
def ecg_r(data, sample_freq, f_base):
    """
    和上边写的心音识别一致，只不过没有那么麻烦
    心电的周期识别起来较为准确，所以其实后边做同步信号处理时，就直接用心电的周期频率代表心电心音同步信号的周期频率完全可以，且这样更准确，因为利用心音信号识别心音周期通常不太准确。
    虽然肉眼看起来很容易看出周期，深度学习算法也很容易识别，但是使用传统的算法提取时我遇到的困难还是挺多的，所以这里可以给出后续做特征工程的一些思路：
    利用心电进行周期的确定，进而对心电信号进行简单的特征提取，重点放在心音特征提取上，但是困难也是很大，心音的杂音太多了，高质量采集应该是先行之道，进而利用数据进行多层次分析，这才能保证可以出一些别人没有做出来的成果。
    """
    # 对心音即使不做包络也可以提取到准确的周期频率，但是一般都做一下，防止意外发生。
    f_base = f_base
    # 窗口
    NT_base = int((1 / f_base) * sample_freq)
    # 寻找第一个R波
    r = [np.argmax(data[0:int(NT_base * 1.1)])]
    while r[-1] + int(NT_base * 1.1) < len(data):
        # 寻找下一个R波，
        temp = np.argmax(data[r[-1] + int(NT_base / 2):r[-1] + int(NT_base * 1.2)]) + r[-1] + int(NT_base / 2)
        r.append(temp)
    # 返回的数据中去除了第一个R波的第一个和最后一个，注意后续处理同步数据时务必保证同时去除，就是心电去除了开头和结尾的一个周期，那心音也要去除
    # 而我在做特征处理算法的一开始是写的同步，但是现阶段需要很多次的测试，有时候只测试了心音，有时候只测试了心电，这就做不到同步，所以目前我是分开处理的。
    return r[1:-1]


# # 绘制包络曲线，这个函数是最早之前使用的网上的代码加自己的修改得来的，过于繁琐，效率不高，所以可以自己看看，但是不能用于项目。
# #输入信号序列即可(list)
# def envelope_extraction(signal):
#     # 转变输入信号的数值类型，必须为float
#     s = signal.astype(float)
#     #
#     q_u = np.zeros(s.shape)
#     q_l =  np.zeros(s.shape)

#     #在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
#     #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
#     u_x = [0,] #上包络的x序列
#     u_y = [s[0],] #上包络的y序列

#     l_x = [0,] #下包络的x序列
#     l_y = [s[0],] #下包络的y序列

#     # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
#     #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

#     for k in range(1,len(s)-1):
#         if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
#             u_x.append(k)
#             u_y.append(s[k])

#         if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
#             l_x.append(k)
#             l_y.append(s[k])

#     u_x.append(len(s)-1) #上包络与原始数据切点x
#     u_y.append(s[-1]) #对应的值

#     l_x.append(len(s)-1) #下包络与原始数据切点x
#     l_y.append(s[-1]) #对应的值

#     #u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
#     upper_envelope_y = np.zeros(len(signal))
#     lower_envelope_y = np.zeros(len(signal))

#     upper_envelope_y[0] = u_y[0]#边界值处理
#     upper_envelope_y[-1] = u_y[-1]
#     lower_envelope_y[0] =  l_y[0]#边界值处理
#     lower_envelope_y[-1] =  l_y[-1]

#     #上包络
#     last_idx,next_idx = 0, 0
#     k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1]) #初始的k,b
#     for e in range(1, len(upper_envelope_y)-1):

#         if e not in u_x:
#             v = k * e + b
#             upper_envelope_y[e] = v
#         else:
#             idx = u_x.index(e)
#             upper_envelope_y[e] = u_y[idx]
#             last_idx = u_x.index(e)
#             next_idx = u_x.index(e) + 1
#             #求连续两个点之间的直线方程
#             k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

#     #下包络
#     last_idx,next_idx = 0, 0
#     k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1]) #初始的k,b
#     for e in range(1, len(lower_envelope_y)-1):

#         if e not in l_x:
#             v = k * e + b
#             lower_envelope_y[e] = v
#         else:
#             idx = l_x.index(e)
#             lower_envelope_y[e] = l_y[idx]
#             last_idx = l_x.index(e)
#             next_idx = l_x.index(e) + 1
#             #求连续两个切点之间的直线方程
#             k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

#     #也可以使用三次样条进行拟合
#     #u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
#     #l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
#     #for k in range(0,len(s)):
#      #   q_u[k] = u_p(k)
#      #   q_l[k] = l_p(k)

#     return upper_envelope_y, lower_envelope_y

# def general_equation(first_x,first_y,second_x,second_y):
#     # 斜截式 y = kx + b
#     A = second_y-first_y
#     B = first_x-second_x
#     C = second_x * first_y - first_x * second_y
#     k = -1 * A / B
#     b = -1 * C / B
#     return k, b


# 寻找心音信号的s1,s2及其间期开始，结束点
def find_s1_s2_lr(data_o, sample_freq):
    """
    data_o:原始数据
    s1:找到的s1波峰点
    s2:找到的s2波峰点
    """
    # S1_L为S1间期的开始点 因为在左边 为left,所以简写叫 S1_L
    S1_L = []
    # S1_R为S1间期的结束点 因为在右边 为right,所以简写叫 S1_R
    S1_R = []
    # s2开始点
    S2_L = []
    # s2结束点
    S2_R = []
    # 对心音信号连续做3次包络，这个3次是我自己试验出来的，效果最好的，后续可以自己试验
    pcg = Envelop(denoise_pcg(data_o), 3)
    # 或者f_base
    f_base, freq1, fft_amp1 = apply_fft_pw(pcg, sample_freq)
    # 这里就是我说的，心音提取的f_base可能会出错，我们这里进行纠正，默认为1hz，也就是1秒钟心脏跳动一次。
    if f_base < 0.5 or f_base > 1.5:
        f_base = 1
    # 窗口
    N_base = int((1 / f_base) * sample_freq)
    # 得到一段心音的S1,S2
    s1, s2 = pcg_s1_s2(data_o, sample_freq, f_base)
    # 这是对原始信号做一次包络的信号，不能直接使用原始信号，因为原始信号存在负值
    data_o3 = Envelop(denoise_pcg2(data_o), 1)
    for i in range(len(s1)):
        # temp参数用来限制窗口大小，这里我们寻找间期的开始点和结束点需要在很小的窗口内寻找。
        temp = 0.08
        S1_L.append(
            np.argmin(data_o3[s1[i] - int(N_base * temp):s1[i] - int(N_base * temp / 4)]) + s1[i] - int(N_base * temp))
        S1_R.append(np.argmin(data_o3[s1[i] + int(N_base * temp / 4):s1[i] + int(N_base * temp)]) + s1[i] + int(
            N_base * temp / 4))
        S2_L.append(
            np.argmin(data_o3[s2[i] - int(N_base * temp):s2[i] - int(N_base * temp / 4)]) + s2[i] - int(N_base * temp))
        S2_R.append(np.argmin(data_o3[s2[i] + int(N_base * temp / 4):s2[i] + int(N_base * temp)]) + s2[i] + int(
            N_base * temp / 4))
    return f_base, s1, s2, S1_L, S1_R, S2_L, S2_R


# 计算s1,s2间期能量
def cal_p_f(data_o, S1_L, S1_R, S2_L, S2_R, sample_freq):
    """
    找到了间期有啥用？当然可以用来计算间期的能量占比了。
    """
    data_test = denoise_pcg3(data_o)
    f = []
    p_f = []
    for i in range(len(S1_L) - 1):
        # 从一个S1的开始点到下一个S1的开始点即为一个周期
        fft_0 = fft(data_test[S1_L[i]:S1_L[i + 1]], sample_freq)
        # 因为fft_0是一次变换，具体的演示可以参考test_draw中的第一个图片
        # 　数学表达式就是　(abs(fft(y)))的平方进而求平均值。
        f0 = np.mean(np.power(np.abs(fft_0[1:int(sample_freq / 2)]), 2))
        # f0是对整个间期做fft变换得到的结果，而f1仅仅是s1间期，下面一样，
        fft_1 = fft(data_test[S1_L[i]:S1_R[i]], sample_freq)
        f1 = np.mean(np.power(np.abs(fft_1[1:int(sample_freq / 2)]), 2))
        fft_2 = fft(data_test[S2_L[i]:S2_R[i]], sample_freq)
        f2 = np.mean(np.power(np.abs(fft_2[1:int(sample_freq / 2)]), 2))
        fft_3 = fft(data_test[S1_R[i]:S2_L[i]], sample_freq)
        f3 = np.mean(np.power(np.abs(fft_3[1:int(sample_freq / 2)]), 2))
        fft_4 = fft(data_test[S2_R[i]:S1_L[i + 1]], sample_freq)
        f4 = np.mean(np.power(np.abs(fft_4[1:int(sample_freq / 2)]), 2))
        f.append([f0, f1 + f2 + f3 + f4, f1, f2, f3, f4])
    for i in range(4):
        # 返回的是f1,f2,f3,f4分别占整个周期的能量比
        p_f.append(np.mean((np.array(f).T[i + 2]) / (np.array(f).T[1])))
    return p_f


def find_r_pt_lr(sample_freq, data_o):
    """
    函数功能：寻找心电的R波，T波，P波并对T波，P波进行波谷的寻找
    data_o：送入进来的数据，可以是原始数据，也可以是降采样完的数据
    sample_freq：当前数据的采样率
    """
    # 存储P波波峰
    P_peak = []
    # P波波谷的左边
    PL_valley = []
    # P波波谷的右边
    PR_valley = []
    # T波波峰
    T_peak = []
    # T波波谷的左边
    TL_valley = []
    # T波波谷的右边
    TR_valley = []
    # 进行3次包络
    ecg = Envelop(denoise_ecg(data_o), 3)
    # 做fft变换
    f_base, freq1, fft_amp1 = apply_fft_pw(ecg, sample_freq)
    # 防止算法寻找到的频率出错，这里需要进行校正，一般来说心电信号只要能看，都不会出错。
    if f_base < 0.7 or f_base > 1.5:
        f_base = random.uniform(1, 1.5)
    # 窗口大小
    N_base = int((1 / f_base) * sample_freq)
    # 寻找波峰 R_peak
    R_peak = ecg_r(data_o, sample_freq, f_base)
    # 　使用滤波后的数据，提升识别准确性
    data_o2 = denoise_ecg2(data_o)
    for i in range(len(R_peak)):
        # P波，还是在R波左边通过划一个小窗口进行寻找  R_peak[i]-int(N_base*0.05) 这个是很重要的， 如果没有-int(N_base*0.05)，那就会找到R波
        temp1 = np.argmax(data_o2[R_peak[i] - int(N_base * 0.2):R_peak[i] - int(N_base * 0.05)]) + R_peak[i] - int(
            N_base * 0.2)
        # 确认好P波位置
        P_peak.append(temp1)
        # 寻找P的间期开始点和结束点
        temp2 = np.argmin(data_o2[P_peak[-1] - int(N_base * 0.1):P_peak[-1]]) + P_peak[-1] - int(N_base * 0.1)
        PL_valley.append(temp2)
        temp3 = np.argmin(data_o2[P_peak[-1]:P_peak[-1] + int(N_base * 0.1)]) + P_peak[-1]
        PR_valley.append(temp3)
        # T波
        temp1 = np.argmax(data_o2[R_peak[i] + int(N_base * 0.15):R_peak[i] + int(N_base * 0.4)]) + R_peak[i] + int(
            N_base * 0.15)
        T_peak.append(temp1)
        temp2 = np.argmin(data_o2[T_peak[-1] - int(N_base * 0.2):T_peak[-1]]) + T_peak[-1] - int(N_base * 0.2)
        TL_valley.append(temp2)
        temp3 = np.argmin(data_o2[T_peak[-1]:T_peak[-1] + int(N_base * 0.2)]) + T_peak[-1]
        TR_valley.append(temp3)
    return f_base, R_peak, P_peak, PL_valley, PR_valley, T_peak, TL_valley, TR_valley


def drawing(data_1, data_2, path, time_id, sample_rate):
    """
    函数功能：产生各类图形及需要展示的各个数据
    data_1：心音数据
    data_2：心电数据
    path：数据保存路径
    time_id：文件名，此处应该为数据文件名的第一个时间戳 ，回忆deta_processing.py文件中的关于时间戳变量：inv_str
    F:.....\\upload_data\\20230423155429_HEART_SOUND_SPP_向梦辉_heart.csv'，我们需要提取的是文件名的20230423155429，这个是时间戳

    执行整个代码，测试耗时大概1.59s，主要耗时在保存csv的文件上
    """
    # 复制一份数据  pcg心音 ecg心电
    data_pcg = data_1
    data_ecg = data_2


    # 只要数据的第二列，数据的第一列均为x轴顺序片[0,1,2,....,n],第二列是真实的数据
    # list_pcg = data_pcg[1].values.tolist()
    # list_ecg = data_ecg[1].values.tolist()
    list_pcg = data_1
    list_ecg = data_2

    # print("type(list_pcg)", type(list_pcg)) #
    # print("type(list_ecg)", type(list_ecg))
    # with open('output.txt', mode='w', encoding='utf-8') as file:
    #     file.write(f"data_1:{data_1}")
    #     file.write(f"data_1:{data_1}")
    #     file.write(f"list_pcg:{list_pcg}")
    #     file.write(f"list_ecg:{list_ecg}")

    data_pcg_1k = []
    data_ecg_1k = []
    if sample_rate == 8000:
        for index in range(min(len(list_pcg), len(list_ecg))):
            if (index % 8 == 0):
                data_pcg_1k.append(list_pcg[index])
                data_ecg_1k.append(list_ecg[index])
    else:
        data_pcg_1k = list_pcg
        data_ecg_1k = list_ecg
    edge = 0.00001
    # 最终进行处理时不管是手持式8k采样率的数据还是穿戴式1k采样率的数据，我们都会变成1k进行处理，因为采样率越高，处理时间越长，响应时间越长，用户友好性越差。
    sr = 1000
    # fft参数的窗口大小，这个值不是很重要，但是他会影响变换后的图像呈现方式，具体到语谱图上，当nfft越小时，图像的周期性越强，频率性越弱，当nfft越大时，图像的周期性越弱，频率性越强，图像可以查看test_draw文件
    nfft = 256
    # 数据的方差过小是绝对不可以的，方差过小只能说明是错误的数据
    if np.var(data_ecg_1k) < edge:
        print("心电数据采集无效")
    if np.var(data_pcg_1k) < edge:
        print("心音数据采集无效")
    if np.var(data_ecg_1k) < edge or np.var(data_pcg_1k) < edge:
        return 0
    else:
        # # 在图像路径下新建以当前time_id命名的文件夹
        # path = os.path.join(path, time_id)
        # # 如果dir_a是一个存在的文件夹，就将该文件夹删除并重新建立一个文件夹
        # if os.path.isdir(path):
        #     # 删除，这是文件树删除，不清楚百度
        #     shutil.rmtree(path, ignore_errors=True)
        # # 创建文件夹
        # os.makedirs(path)

        # 对心电数据操作
        f_base, R_peak, P_peak, PL_valley, PR_valley, T_peak, TL_valley, TR_valley = find_r_pt_lr(sr, data_ecg_1k)
        # print('1',f_base, R_peak, P_peak, PL_valley, PR_valley, T_peak, TL_valley, TR_valley)
        # P-T间期
        temp_pt = np.array(T_peak) - np.array(P_peak)
        # R-R间期
        temp_rr = np.array(R_peak[1:]) - np.array(R_peak[:-1])
        # P波间期
        temp_p = np.array(PR_valley) - np.array(PL_valley)
        # T波间期
        temp_t = np.array(TR_valley) - np.array(TL_valley)

        # # 绘图的尺寸
        # plt.figure(figsize=(4, 3))
        # # PT间期周期图
        # plt.scatter(np.arange(0, len(temp_pt)), temp_pt)
        # plt.ylim(np.min(temp_pt[1:]) - 100, np.max(temp_pt[1:]) + 100)
        # plt.title('PT_temp')
        # plt.savefig(path + "//pt.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # PT间期散点图
        # plt.scatter(temp_pt[:-1], temp_pt[1:])
        # plt.xlim(np.min(temp_pt[:-1]) - 100, np.max(temp_pt[:-1]) + 100)
        # plt.ylim(np.min(temp_pt[1:]) - 100, np.max(temp_pt[1:]) + 100)
        # plt.title('PT_scatter')
        # plt.savefig(path + "//pt_temp.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # RR间期散点图
        # plt.scatter(temp_rr[:-1], temp_rr[1:])
        # plt.xlim(np.min(temp_rr[:-1]) - 300, np.max(temp_rr[:-1]) + 300)
        # plt.ylim(np.min(temp_rr[1:]) - 300, np.max(temp_rr[1:]) + 300)
        # plt.title('RR_scatter')
        # plt.savefig(path + "//rr_temp.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # 画语谱图
        # _ = plt.specgram(data_ecg_1k, NFFT=nfft, Fs=sr)
        # plt.ylabel('Frequency')
        # plt.xlabel('Time(s)')
        # plt.title('ECG_Spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(path + "//spe_ecg.jpg", dpi=300, bbox_inches='tight')
        # plt.close()
        # 计算当前数据
        # p波持续时间（每一段数据中所有P波持续时间的平均值，下同）
        p_temp = float(Decimal(np.mean(temp_p) / sr * 1000).quantize(Decimal("0.0")))
        # t波持续时间
        t_temp = float(Decimal(np.mean(temp_t) / sr * 1000).quantize(Decimal("0.0")))
        # P波，T波间期
        pt_temp = float(Decimal(np.mean(temp_pt) / sr * 1000).quantize(Decimal("0.0")))
        # RR间期
        rr_temp = float(Decimal(np.mean(temp_rr) / sr * 1000).quantize(Decimal("0.0")))
        # 心率
        xl = round(f_base * 60, 0)

        # print(data_pcg_1k)

        # 对心音数据进行操作
        f_base, s1, s2, S1_L, S1_R, S2_L, S2_R = find_s1_s2_lr(data_pcg_1k, sr)
        # print('2',f_base, s1, s2, S1_L, S1_R, S2_L, S2_R)
        temp_s1s1 = np.array(s1)[1:] - np.array(s1)[:-1]
        temp_s2s2 = np.array(s2)[1:] - np.array(s2)[:-1]
        temp_s1s2 = np.array(s2) - np.array(s1)
        temp_s1 = np.array(S1_R) - np.array(S1_L)
        temp_s2 = np.array(S2_R) - np.array(S2_L)
        # 收缩期间期计算
        temp_systole = np.array(S2_L) - np.array(S1_R)
        # 舒张期间期计算
        temp_diastole = np.array(S1_L)[1:] - np.array(S2_R)[:-1]

        # plt.figure(figsize=(4, 3))
        # # s1s2间期周期图
        # plt.scatter(np.arange(0, len(temp_s1s2)), temp_s1s2)
        # plt.ylim(np.min(temp_s1s2[1:]) - 100, np.max(temp_s1s2[1:]) + 100)
        # plt.title('S1S2_temp')
        # plt.savefig(path + "//s1s2.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # s1s2间期散点图
        # plt.scatter(temp_s1s2[:-1], temp_s1s2[1:])
        # plt.xlim(np.min(temp_s1s2[:-1]) - 100, np.max(temp_s1s2[:-1]) + 100)
        # plt.ylim(np.min(temp_s1s2[1:]) - 100, np.max(temp_s1s2[1:]) + 100)
        # plt.title('S1S2_scatter')
        # plt.savefig(path + "//s1s2_temp.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # s1s1间期散点图
        # plt.scatter(temp_s1s1[:-1], temp_s1s1[1:])
        # plt.xlim(np.min(temp_s1s1[:-1]) - 300, np.max(temp_s1s1[:-1]) + 300)
        # plt.ylim(np.min(temp_s1s1[1:]) - 300, np.max(temp_s1s1[1:]) + 300)
        # plt.title('S1S1_scatter')
        # plt.savefig(path + "//s1s1_temp.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # s2s2间期散点图
        # plt.scatter(temp_s2s2[:-1], temp_s2s2[1:])
        # plt.xlim(np.min(temp_s2s2[:-1]) - 300, np.max(temp_s2s2[:-1]) + 300)
        # plt.ylim(np.min(temp_s2s2[1:]) - 300, np.max(temp_s2s2[1:]) + 300)
        # plt.title('S2S2_scatter')
        # plt.savefig(path + "//s2s2_temp.jpg", dpi=300, bbox_inches='tight')
        # plt.clf()
        # # 画语谱图
        # _ = plt.specgram(data_pcg_1k, NFFT=nfft, Fs=sr)
        # plt.ylabel('Frequency')
        # plt.xlabel('Time(s)')
        # plt.title('PCG_Spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(path + "//spe_pcg.jpg", dpi=300, bbox_inches='tight')
        # plt.close()

        # 计算当前数据
        # s1-s1间期
        s1s1_temp = float(Decimal(np.mean(temp_s1s1) / sr * 1000).quantize(Decimal("0.0")))
        # s2-s2间期
        s2s2_temp = float(Decimal(np.mean(temp_s2s2) / sr * 1000).quantize(Decimal("0.0")))
        # s1，s2间期
        s1s2_temp = float(Decimal(np.mean(temp_s1s2) / sr * 1000).quantize(Decimal("0.0")))
        # s1间期
        s1_temp = float(Decimal(np.mean(temp_s1) / sr * 1000).quantize(Decimal("0.0")))
        # s2间期
        s2_temp = float(Decimal(np.mean(temp_s2) / sr * 1000).quantize(Decimal("0.0")))
        # 收缩期
        systole_temp = float(Decimal(np.mean(temp_systole) / sr * 1000).quantize(Decimal("0.0")))
        # 舒张期
        diastole_temp = float(Decimal(np.mean(temp_diastole) / sr * 1000).quantize(Decimal("0.0")))
        # 能量占比
        p_f = cal_p_f(data_pcg_1k, S1_L, S1_R, S2_L, S2_R, sr)
        # s1能量占比
        s1_energy = float(Decimal(p_f[0] * 100).quantize(Decimal("0.0")))
        # s2能量占比
        s2_energy = float(Decimal(p_f[1] * 100).quantize(Decimal("0.0")))
        # 收缩期能量占比
        systole_energy = float(Decimal(p_f[2] * 100).quantize(Decimal("0.0")))
        # 舒张期能量占比
        diastole_energy = float(Decimal(p_f[3] * 100).quantize(Decimal("0.0")))

        # 计算各频段能量占比，这块的主要思路就是利用小波方法进行各个频段的近似分解，具体小波基函数，小波分解层数，小波方法自己都可以改。
        wavelet = "db5"
        level = 5
        data_ps = denoise_pcg3(data_pcg_1k)
        # 简单的归一化[-1,1]
        data_s = np.array(data_ps) / np.max(np.abs(np.array(data_ps)))
        re = pywt.wavedec(data=data_s, wavelet=wavelet, level=level)
        # cA5, cD5, cD4, cD3, cD2, cD1  频率由低到高  对应的分别是 0-15,15-31,31-62,62-125,125-250,250-500.
        energy = []
        for i in re:
            # 二阶范式，本质就是平方和开根号，用循环计算每个频段的能量
            energy.append(pow(np.linalg.norm(i, ord=None), 2))
        energy = np.array(energy)
        # 计算能量占比
        energy_per = np.array(
            [float(Decimal(i / np.sum(np.array(energy)) * 100).quantize(Decimal("0.0"))) for i in energy])

        # 计算联合数据 这个用于同步数据计算，后续可能能用到
        # R到S1时间
        # 计算两者长度差
        length_difference = len(s1) - len(R_peak)
        # 如果 S1 比 R_peak 长，截取 S1 或者填充 R_peak
        if length_difference > 0:
            R_peak = np.pad(R_peak, (0, length_difference), mode='constant', constant_values=np.nan)  # 填充 NaN 或其他值
        elif length_difference < 0:
            s1 = np.pad(s1, (0, -length_difference), mode='constant', constant_values=np.nan)

        temp_rs1 = np.array(s1) - np.array(R_peak)
        rs1_temp = round(np.abs(np.mean(temp_rs1) / sr) * 1000, 2)
        # R到S2时间
        length_difference2 = len(s2) - len(R_peak)
        # 如果 S1 比 R_peak 长，截取 S1 或者填充 R_peak
        if length_difference2 > 0:
            R_peak = np.pad(R_peak, (0, length_difference2), mode='constant', constant_values=np.nan)  # 填充 NaN 或其他值
        elif length_difference2 < 0:
            s2 = np.pad(s2, (0, -length_difference2), mode='constant', constant_values=np.nan)
        temp_rs2 = np.array(s2) - np.array(R_peak)
        rs2_temp = round(np.abs(np.mean(temp_rs2) / sr) * 1000, 2)

        # 保存csv
        # path_save = path + "//show.csv"
        # #         index = ["xl","rr_temp","pt_temp", "t_temp", "p_temp","s1s1_temp","s2s2_temp","s1s2_temp","s1_temp","s2_temp",
        # #                  "systole_temp","diastole_temp","s1_energy","s2_energy","systole_energy","diastole_energy","rs1_temp","rs2_temp"]
        # index = ["xl", "rr_temp", "pt_temp", "t_temp", "p_temp", "s1s1_temp", "s2s2_temp", "s1s2_temp", "s1_temp",
        #          "s2_temp",
        #          "systole_temp", "diastole_temp", "s1_energy", "s2_energy", "systole_energy", "diastole_energy",
        #          "0-15hz",
        #          "15-31hz", "31-62hz", "62-125hz", "125-250hz", "250-500hz", "rs1_temp", "rs2_temp"]
        # # 保存csv文件到对应文件夹
        # #         DF = pd.DataFrame([xl,rr_temp,pt_temp, t_temp, p_temp,s1s1_temp,s2s2_temp,s1s2_temp,s1_temp,s2_temp,
        # #                            systole_temp,diastole_temp,s1_energy,s2_energy,systole_energy,diastole_energy,rs1_temp,rs2_temp],index = index)
        # DF = pd.DataFrame([xl, rr_temp, pt_temp, t_temp, p_temp, s1s1_temp, s2s2_temp, s1s2_temp, s1_temp, s2_temp,
        #                    systole_temp, diastole_temp, s1_energy, s2_energy, systole_energy, diastole_energy,
        #                    energy_per[0], energy_per[1], energy_per[2], energy_per[3], energy_per[4], energy_per[5],
        #                    rs1_temp, rs2_temp], index=index)
        # DF.to_csv(path_save, header=None)
        data_dict = {
            "xl": xl,
            "rr_temp": rr_temp,
            "pt_temp": pt_temp,
            "t_temp": t_temp,
            "p_temp": p_temp,
            "s1s1_temp": s1s1_temp,
            "s2s2_temp": s2s2_temp,
            "s1s2_temp": s1s2_temp,
            "s1_temp": s1_temp,
            "s2_temp": s2_temp,
            "systole_temp": systole_temp,
            "diastole_temp": diastole_temp,
            "s1_energy": s1_energy,
            "s2_energy": s2_energy,
            "systole_energy": systole_energy,
            "diastole_energy": diastole_energy,
            "0-15hz": energy_per[0],
            "15-31hz": energy_per[1],
            "31-62hz": energy_per[2],
            "62-125hz": energy_per[3],
            "125-250hz": energy_per[4],
            "250-500hz": energy_per[5],
            "rs1_temp": rs1_temp,
            "rs2_temp": rs2_temp
        }
        return data_dict
