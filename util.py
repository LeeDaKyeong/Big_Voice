
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence


# call audio with librosa
def call_audio_librosa(path, sr = 44100):
    y, sr = librosa.load(path,sr = sr)
    y = audio_regul(y)
    #y = audio_extract(y)
    return (y, sr)

# call audio with AudioSegment
def call_audio_AudioSegment(path):
    y = AudioSegment.from_file(path)
    return y

# 정규화
def audio_regul(y):
    _y = librosa.util.normalize(y)
    return _y

def audio_extract(y):
    pass

# AudioSegment to librosa
def AudioSegment2librosa(y):
    samples = y.get_array_of_samples()
    samples = np.array(samples)
    samples = audio_regul(samples) #음성 사이즈는 살짝 이상해져서 정규화 필요
    return samples

# librosa to AudioSegment
# 살짝 이상
def librosa2AudioSegment(y, sr = 44100):
    samples = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)
    return samples

# sentece to words,
# input : AudioSegment, output : words list
def word_seperation(y):
    audio_chunks = split_on_silence(y, min_silence_len=30, silence_thresh=-70)
    return audio_chunks

# MFCC extract
# input : librosa, output : numpy
def MFCC_extract(y, sr = 44100, y_len = 50000):
    if len(y) < y_len:
        y = np.append(y,np.zeros(y_len-len(y)))
    else:
        y = y[:y_len]

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


# MFCC extract
# speecker recog 에서 한줄로 핀 mfcc필요
def MFCC_extract_reshape(y, sr = 44100, y_len = 50000):
    if len(y) < y_len:
        y = np.append(y,np.zeros(y_len-len(y)))
    else:
        y = y[:y_len]
    log_S = MFCC_extract(y, sr)
    log_S_reshape = np.reshape(log_S,(log_S.shape[0]*log_S.shape[1]))
    return log_S_reshape
