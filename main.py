import speaker_recog
import speech_recog_API
import matplotlib.pyplot as plt
import util
import numpy as np
import os
import re
import IPython.display as ipd
import librosa
import pyglet
import pygame
import time


if __name__ == "__main__":

    # audio_path = "./TK4.wav"
    # speaker = speaker_recog.speaker_recog(audio_path)
    # print(speaker)


    # LS_y, sr = librosa.load("./test_audio/LS.wav")
    # KY_y, sr = librosa.load("./test_audio/KY.wav")
    # DK_y, sr = librosa.load("./test_audio/DK.wav")
    # HJ_y, sr = librosa.load("./test_audio/HJ.wav")
    # TK_y, sr = librosa.load("./test_audio/TK.wav")
    #
    # y = np.append(LS_y[1000:], KY_y[1000:])
    # y = np.append(y,DK_y[1000:])
    # y = np.append(y,HJ_y[1000:])
    # y = np.append(y,TK_y[1000:])
    #
    # librosa.output.write_wav('testtest.wav', y, sr)

    #pygame.init()
    #pygame.mixer.init()
    #sounda= pygame.mixer.Sound("testtest.wav")

    #sounda.play()
    #time.sleep (10)

    #audio = util.call_audio_AudioSegment('testtest.wav')
    #chunk = util.word_seperation(audio)

    li = os.listdir("./test")
    #print(li)
    li = sorted(li, key = lambda x: (int(re.sub('[^0-9]','',x)),x))

    for i in li:
        if 'chunk' in i:
            #print(os.path.join("./test",i))
            speaker = speaker_recog.speaker_recog(os.path.join("./test",i))
            speech = speech_recog_API.speech_recog_google(os.path.join("./test", i))
            print("speaker : ",speaker, "speech : ",speech)


    #print(speaker)

    #y,sr = util.call_audio_librosa(audio_path)

    #plt.plot(y)
    #plt.show()

    #speech = speech_recog_API.speech_recog_google(audio_path)
    #print(speech)

    #print(speech_recog_API.speech_recog_mic())
