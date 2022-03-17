#######################이건 함수 테스트 하는부분
import librosa
import pyaudio  # 마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import *
import os

##### 변수 설정 부분 #####
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5  # 녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
DATA_PATH = "C:/Users/JISU/OneDrive/바탕 화면/data/data"
#DATA_PATH = "C:/Users/JISU/OneDrive/바탕 화면/프종/프종/data/남자"
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
####################################################


############################# // ac_score // #######################################

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


                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr * 0.8), n_fft=int(sr * 1)).T
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

    # 3d to 2d
    #     nsamples, nx, ny = np.shape(X_train)
    #     X_train = np.reshape(X_train,(nsamples,nx*ny))
    #     nsamples, nx, ny = np.shape(X_test)
    #     X_test = np.reshape(X_test,(nsamples,nx*ny))

    #     Y_train = np.argmax(Y_train, axis=1)###one-hot을 합침
    #     Y_test = np.argmax(Y_test, axis=1)###one-hot을 합침
    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./data.npy", xy)
    # print(X_data)
    # print(Y_label)


load_wave_generator(DATA_PATH)




########################################## 음성 녹음 부분 ##################



########################## 논리 회귀 ###########################
clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))

#y, sr = librosa.load("./youinna16.wav")
#y, sr = librosa.load("./test_홍지수3.wav")
y, sr = librosa.load("./output.wav")


mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr*0.08),n_fft=int(sr*1)).T
#print(mfcc.shape)
#y=librosa.load("./output.wav",16000)[0]
#S=librosa.feature.melspectrogram(y=y,n_mels=80,n_fft=512,win_length=400,hop_length=160)#320/80
#a=feature_extraction("./output.wav")

y_test_estimated = clf.predict(mfcc)




# 음성 입력 후 그것에 대한 정답률 구하기
input_ac_score = metrics.accuracy_score(np.full(len(mfcc),0), y_test_estimated)
print("정답률 =", input_ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))



###########################################
mfcc2= librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
y_test_estimated2 = clf.predict(mfcc2)
input_ac_score2 = metrics.accuracy_score(np.full(len(mfcc2),0), y_test_estimated2)
print("정답률 =", input_ac_score2)

mfcc3= librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr*0.8),n_fft=int(sr*1)).T
y_test_estimated3 = clf.predict(mfcc3)
input_ac_score3 = metrics.accuracy_score(np.full(len(mfcc3),0), y_test_estimated3)
print("정답률 =", input_ac_score3)

mfcc4= librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr*0.45),n_fft=int(sr*0.6)).T
y_test_estimated4 = clf.predict(mfcc4)
input_ac_score4 = metrics.accuracy_score(np.full(len(mfcc4),0), y_test_estimated4)
print("정답률 =", input_ac_score4)


mfcc5= librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr*0.85),n_fft=int(sr*1.1)).T
y_test_estimated5 = clf.predict(mfcc5)
input_ac_score5 = metrics.accuracy_score(np.full(len(mfcc5),0), y_test_estimated5)
print("정답률 =", input_ac_score5)




#################################### 그래프 이미지 #############33

import matplotlib.pyplot as plt

data = [ac_score,input_ac_score,input_ac_score2,input_ac_score3,input_ac_score4,input_ac_score5]
# 네 옵티마이저의 정확률을 박스플롯으로 비교
plt.boxplot(data,labels=["ac_score","1","2","3","4","5"])
plt.title("Accuracy Comparison")
plt.grid()
plt.show()

#import matplotlib.pyplot as plt

#data = [ac_score],[ac_score2],[ac_score3],[ac_score4]
# 네 옵티마이저의 정확률을 박스플롯으로 비교
#plt.boxplot(data,labels=["ac_score","ac_Score2","ac_score3","ac_score4"])
#plt.title("Accuracy Comparison")
#plt.grid()
#plt.show()