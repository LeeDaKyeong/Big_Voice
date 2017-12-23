import util
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import load_model

def data_generate():
    path = './audio_date'  # Put your input directory
    date = ['11.30', '12.1', '12.2', '12.3', '12.4', '12.5', '12.6', '12.7', '12.8']
    person = ['DK', 'HJ', 'KY', 'LS', 'TK']
    word = ['word1', 'word2', 'word3']

    data = np.zeros((1, 128, 196))
    Y = list()

    for i_date in date:
        path1 = os.path.join(path, i_date)

        for i_person in person:
            path2 = os.path.join(path1, i_person + '_' + i_date)

            for i in os.listdir(path2):
                path3 = os.path.join(path2, i)

                for i_word in word:

                    if i_word in path3:
                        for j in os.listdir(path3):
                            path4 = os.path.join(path3, j)

                            y, sr = util.call_audio_librosa(path4)
                            mfcc = util.MFCC_extract(y)
                            _mfcc = np.reshape(mfcc, (1, 128, 196))

                            data = np.vstack((data, _mfcc))
                            Y.append(i_date + i_word)

    data = data[1:]
    print("data generate end!")

    return (data,Y)

def Y_generate(Y):
    _Y = pd.DataFrame(Y)
    dummy_var = pd.get_dummies(_Y,prefix="")
    _Y = np.array(dummy_var)

    return _Y

def X_generate(data):
    X = data.copy()
    X = X.reshape(X.shape[0], -1, 1)
    X = X.astype('float32')

    return X

def data_split(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    return (x_train, x_test, y_train, y_test)

def modeling(x_train):
    model = Sequential()
    # model.add(SimpleRNN(128,
    #                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
    #                    recurrent_initializer=initializers.Identity(gain=1.0),
    #                    activation='relu',
    #                    input_shape=x_train.shape[1:]))

    model.add(LSTM(128, activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dense(27, activation='softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def model_train(x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=100,
                        verbose=1,
                        validation_data=(x_test, y_test))

    return history

def model_load():
    model = load_model('./model/stt_rnn_mfcc.h5')
    return model

def model_test(path):
    y, sr = util.call_audio_librosa(path)

    mfcc = util.MFCC_extract(y)

    X = mfcc.copy()
    X = X.reshape(1, -1, 1)
    X = X.astype('float32')

    model = model_load()

    return model.predict_classes(X)


if __name__ == "__main__":
    data, Y = data_generate()
    print(data.shape)

    Y = Y_generate(Y)
    X = X_generate(data)
    x_train, x_test, y_train, y_test = data_split(X,Y)

    model = modeling(x_train)
    history = model_train(x_train, x_test, y_train, y_test)

    model.save('./model/stt_rnn_mfcc.h5')

    plt.plot(history.history["loss"])
    plt.title("Loss")
    plt.show()

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('IRNN test score:', scores[0])
    print('IRNN test accuracy:', scores[1])