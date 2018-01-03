import speaker_recog
import speech_recog_API
import matplotlib.pyplot as plt
import util
import numpy as np
import os
import re

if __name__ == "__main__":
    #audio_path = "./HJ.wav"
    audio_path = "./audio_date/12.8/HJ_12.8/HJ_word2/word2_4.wav"
    speaker = speaker_recog.speaker_recog(audio_path)

'''
    audio = util.call_audio_AudioSegment(audio_path)
    chunk = util.word_seperation(audio)

    li = os.listdir("./test")
    #print(li)
    li = sorted(li, key = lambda x: (int(re.sub('[^0-9]','',x)),x))

    for i in li:
        if 'chunk' in i:
            print(os.path.join("./test",i))
            speaker = speaker_recog.speaker_recog(os.path.join("./test",i))
            #print(speaker)
            
'''

    #print(speaker)

    #y,sr = util.call_audio_librosa(audio_path)

    #plt.plot(y)
    #plt.show()

    #speech = speech_recog_API.speech_recog_google(audio_path)
    #print(speech)

    #print(speech_recog_API.speech_recog_mic())
