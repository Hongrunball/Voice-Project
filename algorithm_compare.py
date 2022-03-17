#######################이건 함수 테스트 하는부분
import librosa
import pyaudio  # 마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, validation_curve
from sklearn import *
import os
import time
#import tensorflow as tf
from sklearn import svm

##### 변수 설정 부분 #####
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5  # 녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
#DATA_PATH = "./data20"
DATA_PATH = "C:/Users/JISU/OneDrive/바탕 화면/data/data"
X_train = []  # train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []
####################################### mel-spectogram ################
min_level_db=-100

def normalize_mel(S):
    return np.clip((S-min_level_db/-min_level_db,0,1))

def feature_extraction(path):
    y=librosa.load(path,16000)[0]
    S=librosa.feature.melspectrogram(y=y,n_mels=80,n_fft=512,win_length=400,hop_length=160)#320/80
    norm_log_S=normalize_mel(librosa.power_to_db(S,ref=np.max))
    return norm_log_S
#########################################################################

############################# 폴더에 있는 음성데이터 열고 전처리 #######################################

def compute_mfcc(y,sr):

    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr * 0.8), n_fft=int(sr * 1)).T

    return mfcc_feat
#y,sr = librosa.load("./test_홍지수3.wav")
#mfcc=compute_mfcc(y,sr)


def load_wave_generator(path):
    batch_waves = []
    labels = []
    X_data = []
    Y_label = []
    idx = 0
    global X_train, X_test, Y_train, Y_test
    folders = os.listdir(path)

    for folder in folders:
        if not os.path.isdir(path): continue  # 폴더가 아니면 continue
        files = os.listdir(path + "/" + folder)
        print("Foldername :", folder, "-", len(files))
        # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            else:
                print("Filename :", wav)  # .wav 파일이 아니면 continue
                y,sr= librosa.load(path + "/" + folder + "/" + wav)

                mfcc=compute_mfcc(y,sr)  # compute_mfcc()함수 사용
                X_data.extend(mfcc)
                label = [0 for i in range(len(folders))]
                label[idx] = 1
                for i in range(len(mfcc)):
                    Y_label.append(label)
        idx = idx + 1
    # end loop
    print("X_data :", np.shape(X_data))
    print("Y_label :", np.shape(Y_label))

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_label)

    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./data.npy", xy)

load_wave_generator(DATA_PATH)





################# 알고리즘에 따라 검증을 수행하고 정확률을 반환하는 함수 #################
def cross_validation(num):
    if (num == 0):
        clf = Perceptron(max_iter=500, eta0=0.001, verbose=0)  # 퍼셉트론
        clf.fit(X_train, np.argmax(Y_train, axis=1))
    elif (num == 1):
        clf = MLPClassifier()  # 다층 퍼셉트론
        clf.fit(X_train, np.argmax(Y_train, axis=1))
        prange = range(50, 1001, 50)
        train_score, test_score = validation_curve(clf, X_train, Y_train, param_name="hidden_layer_sizes",
                                                   param_range=prange, cv=10, scoring="accuracy", n_jobs=4)
        train_mean = np.mean(train_score, axis=1)
    elif (num == 2):
        clf = LogisticRegression()  # 논리회귀 법
        clf.fit(X_train, np.argmax(Y_train, axis=1))

    y_test_estimated = clf.predict(X_test)

    conf = np.zeros((30, 30))
    for i in range(len(y_test_estimated)):
        conf[y_test_estimated[i], [Y_test[i]]] += 1
    print(conf)

    no_correct = 0
    for i in range(30):
        no_correct += conf[i][i]
    accuracy = no_correct / len(y_test_estimated)

    # 머신러닝의 정답률 구하기
    ac_score = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
    return ac_score


acc_perceptron = cross_validation(0)
acc_mlpclassifier = cross_validation(1)
acc_logistic = cross_validation(2)

print("Perceptron accuracy: ", acc_perceptron)
print("MLPClassifier accuracy: ", acc_mlpclassifier)
print("Logistic accuracy: ", acc_logistic)

#######################################################################################

###########################################################################3
# 네 알고리즘의 정확률을 박스플롯으로 비교
import matplotlib.pyplot as plt

data = [acc_perceptron],[acc_mlpclassifier],[acc_logistic]
plt.boxplot(data,labels=["Perceptron","MLPClassifier","Logistic"])
plt.title("Algorithm Accuracy Comparison")
plt.grid()
plt.show()
plt.savefig('Algorithm Accuracy Comparison')

