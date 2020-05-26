from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os

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

def calculate_edge_freqs(f_remove, bandwidth, fs):
    """
        returns the lower and higher edge of a filter passband, normalized to half cycles / sample
        (freq / (fs / 2)) 
    """
    low = f_remove - bandwidth/2.0
    high = f_remove + bandwidth/2.0
    
    nyq_freq = fs / 2.0
    low = low / nyq_freq  
    high = high / nyq_freq

    return (low, high)
    
def bandstop_iirfilter(f_remove, bandwidth, order, fs, filter_type='butter'):
    """
        Design an IIR (infinite impulse response) bandstop filter using scipy.signal.iirfilter
        
        :param f_remove: the center frequency of the band
        :param bandwidth: the bandwidth of the filter 
        :param order: the order of the filter
        :param fs: the sampling frequency of the signal 
        :param filter_type: the type of filter, default is 'butter'

        :returns b: the numerator polynomial of the filter
        :returns a: the denominator polynomial of the filter
    """

    low, high = calculate_edge_freqs(f_remove, bandwidth, fs)

    b, a = signal.iirfilter(order, [low, high], btype='bandstop',
                     analog=False, ftype=filter_type)
    return (b, a)

def channels_to_stereo(left_channel, right_channel):
    """
        takes two row vector channels and converts them to a single, stereo signal
    """
    return np.column_stack((left_channel.transpose(), right_channel.transpose()))

def design_bandstop_butter_filter(f_remove, bandwidth, order, fs, sos=True):

    low, high = calculate_edge_freqs(f_remove, bandwidth, fs)
    if sos:
        return signal.butter(order, [low, high], btype='bandstop', output='sos')
    else:
        return signal.butter(order, [low, high], btype='bandstop')

def bandstop_butter(sig, num_poly, den_poly, stereo=True):
    """
        filter a given signal using the polynomials of a filter, and return the filtered signal
        the signal can be either mono or stereo
    """
    if not stereo:
        return signal.filtfilt(num_poly, den_poly, sig)
    
    left_channel = signal.filtfilt(num_poly, den_poly, sig[:, 0])
    right_channel = signal.filtfilt(num_poly, den_poly, sig[:, 1])

    return channels_to_stereo(left_channel, right_channel)

def filter(sig, sosfilter, stereo=True):
    """
        same as bandstop_butter but uses second-order sections (sosfiltfilt), which have 'fewer 
        numerical problems'
    """

    if not stereo:
        return signal.sosfiltfilt(sosfilter, sig)
    
    left_channel = signal.sosfiltfilt(sosfilter, sig[:, 0])
    right_channel = signal.sosfiltfilt(sosfilter, sig[:, 1])

    return channels_to_stereo(left_channel, right_channel)

def design_lowpass_butter_filter(f_cutoff, order, fs, sos=True):
    nyq = fs / 2.0
    f_cutoff /= nyq
    if sos:
        return signal.butter(order, f_cutoff, btype='lowpass', output='sos')
    else:
        return signal.butter(order, f_cutoff, btype='lowpass')

def design_highpass_butter_filter(f_cutoff, order, fs, sos=True):
    nyq = fs / 2.0
    f_cutoff /= nyq
    if sos:
        return signal.butter(order, f_cutoff, btype='highpass', output='sos')
    else:
        return signal.butter(order, f_cutoff, btype='highpass')

def memoize_filters(filt_dict, f_remove, fs, bandwidth, order, ftype):
    if not fs in filt_dict:
        filt_dict[fs] = []
        if ftype == 'bandstop':
            for freq in f_remove:
                filt_dict[fs].append(design_bandstop_butter_filter(freq, bandwidth, order, fs))
        elif ftype == 'lowpass':
            filt_dict[fs].append(design_lowpass_butter_filter(f_remove, order, fs))

def filter_all_files(dirname, f_remove, order=1, new_dirname=None, ftype='bandstop', bandwidth=None):
    """
        f_remove is an array of frequencies to center bandstop filters at, filtering the signal using
        each frequency
    """
    # try to create the directory to save the filtered files in if specified
    if new_dirname: 
        new_dirname = os.path.join(os.getcwd(), new_dirname)
        try:
            os.makedirs(new_dirname)
        except OSError:
            pass
    filters = dict()
    filtered_signals = dict()
    for filename in os.listdir(dirname):
        if not filename.lower().endswith('.WAV'.lower()): # ignore non wav files
            continue
        fs, sig = wavfile.read(os.path.join(dirname, filename))
        memoize_filters(filters, f_remove, fs, bandwidth, order, ftype)

        is_stereo = len(sig.shape) == 2 

        filtered_sig = sig
        for filt in filters[fs]:
            filtered_sig = filter(filtered_sig, filt, stereo=is_stereo)

        filtered_signals[filename] = filtered_sig
        if new_dirname:
            path = os.path.join(new_dirname, filename)
            filtered_sig = np.asarray(filtered_sig, dtype=sig.dtype)
            wavfile.write(path, fs, filtered_sig)
    
    return filtered_signals

def plot_specgrams(data, filt_data, Fs, stereo=True):
    cmap = plt.cm.get_cmap('nipy_spectral')
    figsize = (10, 8)
    NFFT = 1024
    if stereo:
        fig, (ax_top, ax_bot) = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        ax_top[0].specgram(data[:, 0], NFFT=NFFT, Fs=Fs, noverlap=900, cmap=cmap)
        ax_bot[0].specgram(data[:, 1], NFFT=NFFT, Fs=Fs, noverlap=900, cmap=cmap)
        ax_top[0].set_title('Left Channel')
        ax_bot[0].set_title('Right Channel')
        ax_top[0].set_xlabel('Time [s]')
        ax_top[0].set_ylabel('Freq [Hz]')

        s, f, t, c = ax_top[1].specgram(filt_data[:, 0], NFFT=NFFT, Fs=Fs, noverlap=900, cmap=cmap)
        ax_bot[1].specgram(filt_data[:, 1], NFFT=NFFT, Fs=Fs, noverlap=900, cmap=cmap)
        ax_top[1].set_title('Filtered Data')
        fig.colorbar(c)
        plt.show()

if __name__ == '__main__':
    # filter along 1 frequency
    # sigs = filter_all_files('GY01', [4000, 5000, 6000, 7000], 500, 1, new_dirname='filtered_tests')
    # filter_all_files('mono', [120000, 130000], order=1, new_dirname='mono_filtered', ftype='bandstop', bandwidth=2000)

    fs, data = wavfile.read('stereo.wav')
    fs, filtered = wavfile.read('filtered.wav')
    plot_specgrams(data, filtered, fs)
