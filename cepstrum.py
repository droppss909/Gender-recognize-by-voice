import numpy as np

from pydub import AudioSegment
import sys

import soundfile as sf




min_human_f=40
max_human_f=250

def cepstrum(signal, sample_freq):
    frame_size = signal.size
    windowed_signal = np.hamming(frame_size) * signal
    dt = 1/sample_freq
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    signal = np.fft.rfft(windowed_signal)
    log = np.log(np.abs(signal))
    cepstrum = np.fft.rfft(log)
    df = freq_vector[1] - freq_vector[0]
    cepstrum_f = np.fft.rfftfreq(log.size, df)
    return cepstrum_f, cepstrum

def cepstrum_f0_detection(signal, sample_freq):

    f, diwmo = cepstrum(signal, sample_freq)
    diwmo=np.abs(diwmo)
    list_human_f=[]
    for i in range(len(f)):
        if (f[i]<=1/min_human_f and f[i]>=1/max_human_f):
            list_human_f.append(diwmo[i])
        else:
            list_human_f.append(0)
    max_human_f0=np.argmax(np.abs(list_human_f))

    f0=1/f[max_human_f0]
    return f0

if __name__=='__main__':

    signal,sample_freqs = sf.read(sys.argv[1])

    if signal[1].size!=1:
        sound = AudioSegment.from_wav(sys.argv[1])
        sound = sound.set_channels(1)
        sound.export("output.wav", format="wav")
        signal, sample_freqs=sf.read("output.wav")
    f0 = cepstrum_f0_detection(signal, sample_freqs)
    print(f0)
    if f0<165:
        print("M")
    else:
        print("F")
