import util
import os
import numpy as np
from sklearn import svm
import pickle

# speaker recog train data 생성(svm)
def train_data(path):
    li = list()

    for index1 in os.listdir(path):
        path2 = os.path.join(path, index1)  # /Users/apple/Desktop/audio_name/DK/DK_11.30
        for index2 in os.listdir(path2):
            path3 = os.path.join(path2, index2)  # /Users/apple/Desktop/audio_name/DK/DK_11.30/DK_sentence1
            for index3 in os.listdir(path3):
                path4 = os.path.join(path3,
                                     index3)  # /Users/apple/Desktop/audio_name/DK/DK_11.30/DK_sentence1/sentence1_1.wav

                y, sr = util.call_audio_librosa(path4)
                li.append(util.MFCC_extract_reshape(y))
    _li = np.array(li)
    return _li


# svm model load
def speaker_recog_model_load(path = './model/speaker_recog_svm.sav'):
    model = pickle.load(open(path, 'rb'))
    return model

# test
def speaker_recog(audio_path):
    speaker = {0: "다경",
               1: "혜진",
               2: "강열",
               3: "이삭",
               4: "태권"}

    y,sr = util.call_audio_librosa(audio_path)
    mfcc = util.MFCC_extract_reshape(y)
    _mfcc = np.reshape(mfcc, (1, len(mfcc)))
    model = speaker_recog_model_load()
    result = model.predict(_mfcc)
    return speaker[int(result)]


if __name__ == "__main__":
    DK_path = "./audio_name/DK"
    DK = train_data(DK_path)
    DK_len = len(DK)

    HJ_path = "./audio_name/HJ"
    HJ = train_data(HJ_path)
    HJ_len = len(HJ)

    KY_path = "./audio_name/KY"
    KY = train_data(KY_path)
    KY_len = len(KY)

    LS_path = "./audio_name/LS"
    LS = train_data(LS_path)
    LS_len = len(LS)

    TK_path = "./audio_name/TK"
    TK = train_data(TK_path)
    TK_len = len(TK)

    X = np.vstack((DK, HJ, KY, LS, TK))
    Y = np.zeros(DK_len+HJ_len+KY_len+LS_len+TK_len)
    Y[:DK_len] = 0
    Y[DK_len:HJ_len] = 1
    Y[DK_len+HJ_len:DK_len+HJ_len+KY_len] = 2
    Y[DK_len+HJ_len+KY_len:DK_len+HJ_len+KY_len+LS_len] = 3
    Y[DK_len+HJ_len+KY_len+LS_len:] = 4

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)

    filename = './model/speaker_recog_svm.sav'
    pickle.dump(lin_clf, open(filename, 'wb'))
