import librosa

def pitchUp(y, sr = 44100, n_step = 10):
    y_pitch_higher = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_higher

def pitchDown(y, sr = 44100, n_step = -10):
    y_pitch_lower = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_lower

def speedUp(y, n_step = 2):
    y_D = librosa.stft(y)
    y_D_fast = librosa.phase_vocoder(y_D, n_step)
    y_faster = librosa.istft(y_D_fast)
    
    return y_faster


def speedDown(y, n_step=0.5):
    y_D = librosa.stft(y)
    y_D_slow = librosa.phase_vocoder(y_D, n_step)
    y_slower = librosa.istft(y_D_slow)

    return y_slower

