#!/usr/bin/env python3
# Plots slow antenna "raw" files

import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.signal as signal
import sa_common

def notch_sixty(s, fs):
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 2.0  # Quality factor = center / 3dB bandwidth
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Slow Antenna .raw data')
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Path or paths to slow antenna files to plot')
    parser.add_argument(
        '--use-notch-sixty', help='Specify this to use a 60 Hz notch filter', action='store_true')
    parser.add_argument(
        '-o', '--output', help='Output path to save plots. If unspecified, plot will be opened in matplotlib show() window.', default=None)
    args = parser.parse_args()
    filenames = sorted(args.input)
    all_data = []
    all_times = []
    for i, filename in enumerate(filenames):
        if i == 0:
            this_file = sa_common.read_SA_file(filename)
        else:
            this_file = sa_common.read_SA_file(filename, previous_file=filenames[i-1])
        this_times, this_data = sa_common.decode_SA_array(this_file)
        all_times.append(this_times)
        all_data.append(this_data)
    adc = np.concatenate(all_data, axis=0)
    adc_ready = np.concatenate(all_times, axis=0)
    adc_ready = sa_common.correct_micros(np.concatenate(all_times))
    delta_t_adc = (adc_ready[-1]-adc_ready[0])*1e-6
    sample_rate = 1.0e6/np.median(np.ediff1d(adc_ready))
    print(
        f"Elapsed time {delta_t_adc:6.3} s with sample rate {sample_rate:6.1f} Hz")

    t = np.arange(adc.shape[0])/sample_rate
    if args.use_notch_sixty:
        adc_filt = notch_sixty(adc, sample_rate)
    else:
        adc_filt = adc

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 6.5))
    fig.suptitle('; '.join(filenames))

    axs[0, 0].specgram(adc-adc.mean(), Fs=sample_rate)
    axs[0, 0].set_title('Spectrogram (raw)')
    axs[0, 0].set_ylabel('Frequency (Hz)')
    axs[0, 0].set_ylim(0, 360)

    axs[1, 0].specgram(adc_filt-adc_filt.mean(),
                       Fs=sample_rate, window=np.blackman(256))
    axs[1, 0].set_title('Spectrogram (filtered)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    axs[1, 0].set_ylim(0, 360)
    axs[1, 0].set_xlabel('Time (s)')

    axs[0, 1].plot(t, adc_ready-adc_ready[0])
    axs[0, 1].set_title('ADC ready microsecond')

    bits_to_volts = (5/((2**24)-1))
    axs[1, 1].plot(t, adc_filt*bits_to_volts)  # -adc_filt.mean())
    axs[1, 1].set_title('ADC (filtered)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylim([-0.6, 2.6])

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output)
