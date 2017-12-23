import speaker_recog
import speech_recog_API
import matplotlib.pyplot as plt
import util
import numpy as np

if __name__ == "__main__":
    audio_path = "./audio_date/11.30/LS_11.30/LS_sentence1/sentence1_2.wav"

    #speaker = speaker_recog.speaker_recog(audio_path)
    #print(speaker)

    y,sr = util.call_audio_librosa(audio_path)

    plt.plot(y)
    plt.show()
    #speech = speech_recog_API.speech_recog_google(audio_path)
    #print(speech)

    #print(speech_recog_API.speech_recog_mic())
