from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def bandstop(f1_edge, f2_edge, window, order):
    if not window:
        window = 'hamming'
    filter_order = order
    fs = 192e3 # sampling rate of data
    fir_coefficients = signal.firwin(filter_order + 1, [f1_edge, f2_edge], window=window, pass_zero='bandstop', fs=fs)
    return fir_coefficients;

def plot_fir_response(fir_coefficients, amplitude=None):
    freqs, response = signal.freqz(fir_coefficients)
    fig, ax1 = plt.subplots()
    ax1.set_title('Frequency Response')
    ax1.set_xlabel('Frequency [rad/sample]') # f[radians/sample] = f[cycles/sec] * (2pi/fs)
    if amplitude == 'dB':
        ax1.plot(freqs, 20 * np.log10(abs(response)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
    else:
        ax1.plot(freqs, response, 'b')
        ax1.set_ylabel('Amplitude', color='b')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(response))
    ax2.plot(freqs, angles, 'r')
    ax2.set_ylabel('Angle (radians)', color='r')
    ax2.grid()
    ax2.axis('tight')
    plt.show()


fir = bandstop(4750, 5250, None, 1)
amp = None
plot_fir_response(fir, amplitude=amp)


