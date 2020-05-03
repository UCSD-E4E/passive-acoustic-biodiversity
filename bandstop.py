from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def bandstop_firwin(f1_edge, f2_edge, order, fs):
    """
        Design a FIR (finite impulse response) bandstop filter using scipy.signal.firwin
        
        :param f1_edge: the lower frequency band of the filter
        :param f2_edge: the the upper frequeny band of the filter
        :param order: the order of the filter, must be even
        :param fs: the sampling frequency of the signal 

        :returns: the coefficients of the the FIR filter, to be convolved with time-domain signal
    """

    fir_coefficients = signal.firwin(order + 1, [f1_edge, f2_edge], window='hamming', pass_zero='bandstop', fs=fs)
    return fir_coefficients

def bandstop_iirnotch(f_remove, bandwidth, fs):
    """
        Design an IIR (infinite impulse response) notch filter using scipy.signal.iirnotch
        
        :param f_remove: the center frequency of the band
        :param bandwidth: the bandwidth of the filter 
            (low = f_remove - (bandwidth / 2))
            (high = f_remove + (bandwidth / 2))
        :param fs: the sampling frequency of the signal 

        :returns b: the numerator polynomial of the filter
        :returns a: the denominator polynomial of the filter
    """
    quality_factor = fs / bandwidth # f_remove / bandwidth 
    b, a = signal.iirnotch(f_remove, quality_factor, fs=fs)
    return (b, a)
    
def plot_freq_response(num_poly, den_poly, fs):
    freq, response = signal.freqz(num_poly, den_poly, fs=fs)
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(response)), color='b')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='b')
    ax[0].grid()
    angles = np.unwrap(np.angle(response)) * 180/np.pi
    ax[1].plot(freq, angles, color='g')
    ax[1].set_ylabel("Angle (degrees)", color='g')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].grid()
    plt.show()

def bandstop_iirfilter(f_remove, bandwidth, order, fs, filter_type='butter'):
    """
        Design an IIR (infinite impulse response) bandstop filter using scipy.signal.iirfilter
        
        :param f_remove: the center frequency of the band
        :param bandwidth: the bandwidth of the filter 
            (low = f_remove - (bandwidth / 2)), normalized according to the nyquist frequency (fs / 2)
            (high = f_remove + (bandwidth / 2)), normalized according to the nyquist frequency (fs / 2)
        :param order: the order of the filter
        :param fs: the sampling frequency of the signal 
        :param filter_type: the type of filter, default is 'butter'

        :returns b: the numerator polynomial of the filter
        :returns a: the denominator polynomial of the filter
    """
    nyq  = fs/2.0
    low  = f_remove - bandwidth/2.0
    high = f_remove + bandwidth/2.0

    # normalize edge frequencies according to (half cycles / sample) 
    # (freq / (fs / 2)) where fs / 2 = nyq
    low  = low/nyq
    high = high/nyq

    b, a = signal.iirfilter(order, [low, high], btype='bandstop',
                     analog=False, ftype=filter_type)
    return (b, a)

def bandstop_butter(f_remove, bandwidth, order, fs):
    """
        Design an IIR (infinite impulse response) butterworth bandstop filter using scipy.signal.butter
        
        :param f_remove: the center frequency of the band
        :param bandwidth: the bandwidth of the filter 
            (low = f_remove - (bandwidth / 2)), normalized according to the nyquist frequency (fs / 2)
            (high = f_remove + (bandwidth / 2)), normalized according to the nyquist frequency (fs / 2)
        :param order: the order of the filter
        :param fs: the sampling frequency of the signal 

        :returns b: the numerator polynomial of the filter
        :returns a: the denominator polynomial of the filter
    """
    nyq  = fs/2.0
    low  = f_remove - bandwidth/2.0
    high = f_remove + bandwidth/2.0

    # normalize edge frequencies according to (half cycles / sample) 
    # (freq / (fs / 2)) where fs / 2 = nyq
    low  = low/nyq
    high = high/nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')
    return (b, a)


if __name__ == '__main__':
    file = 'GY01/trimmed.wav'
    fs, sig= wavfile.read(file)
    order = 1
    f_remove = 5000
    bandwidth = 250

    b, a = bandstop_butter(f_remove, bandwidth, order, fs)
    plot_freq_response(b, a, fs)

    # notch_filtered = signal.lfilter(b, a, sig)
    notch_filtered_left = signal.filtfilt(b, a, sig[:, 0])
    notch_filtered_right = signal.filtfilt(b, a, sig[:, 1])
    filt_sig = np.column_stack((notch_filtered_left.transpose(), notch_filtered_right.transpose()))
    # if you want to write it back as a WAV file
    # filt_sig = np.asarray(filt_sig, dtype=np.int16)
    # wavfile.write('GY01/filtered_notch.WAV', fs, filt_sig)

    figsize = (10, 8)
    fig, (ax_sig, ax_filtered) = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)
    NFFT = 1024

    ax_sig[0].specgram(sig[:, 0], NFFT=NFFT, Fs=fs, noverlap=900)
    ax_sig[0].set_title('Original Signal Left Channel')
    ax_sig[1].specgram(sig[:, 1], NFFT=NFFT, Fs=fs, noverlap=900)
    ax_sig[1].set_title('Original Signal Right Channel')

    ax_filtered[0].specgram(notch_filtered_left, NFFT=NFFT, Fs=fs, noverlap=900)
    ax_filtered[0].set_title('Filtered Signal Left Channel')
    ax_filtered[1].specgram(notch_filtered_right,NFFT=NFFT, Fs=fs, noverlap=900)
    ax_filtered[1].set_title('Filtered Signal Right Channel')
    plt.show()

