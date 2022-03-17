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
DATA_PATH = "C:/Users/JISU/OneDrive/바탕 화면/프종/프종/data"
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


                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T
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

clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))

##########################################  // ac_score2 // ############################################

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


                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T
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
##########################################
######## 음성 데이터를 녹음 해 저장하는 부분 ########
#p = pyaudio.PyAudio()  # 오디오 객체 생성

#stream = p.open(format=FORMAT,  # 16비트 포맷
                #channels=CHANNELS,  # 모노로 마이크 열기
                #rate=RATE,  # 비트레이트
                #input=True,
                #frames_per_buffer=CHUNK)  # CHUNK만큼 버퍼가 쌓인다.

#print("Start to record the audio.")


#frames = []  # 음성 데이터를 채우는 공간

#for i in range(0, int(RATE / CHUNK * 15)):
    # 지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
    #data = stream.read(CHUNK)
    #frames.append(data)

#print("Recording is finished.")

#stream.stop_stream()  # 스트림닫기
#stream.close()  # 스트림 종료
#p.terminate()  # 오디오객체 종료

# WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
#wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#wf.setnchannels(CHANNELS)
#wf.setsampwidth(p.get_sample_size(FORMAT))
#wf.setframerate(RATE)
#wf.writeframes(b''.join(frames))
#wf.close()

#spf = wave.open(WAVE_OUTPUT_FILENAME, 'r')

#signal = spf.readframes(-1)
#signal = np.fromstring(signal, dtype=np.int16)

# 시간 흐름에 따른 그래프를 그리기 위한 부분
#Time = np.linspace(0, len(signal) / RATE, num=len(signal))

#plt.figure(1)
#plt.title('Voice Signal Wave...')
# plt.plot(signal) // 음성 데이터의 그래프
#plt.plot(Time, signal)
#plt.show()



########################## 논리 회귀 ###########################
clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score2 = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score2)
print(pd.value_counts(pd.Series(y_test_estimated)))

############################################################################################################
########################### // ac_Score3 // ##################################################

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

clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score3 = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score3)
print(pd.value_counts(pd.Series(y_test_estimated)))






############################# // ac_score4 // #######################################

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


                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr * 0.9), n_fft=int(sr * 1.1)).T
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

clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score4 = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score4)
print(pd.value_counts(pd.Series(y_test_estimated)))


#####################ac_score5##################################################3

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


                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr * 1.1), n_fft=int(sr * 1.2)).T
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

clf = LogisticRegression() #논리회귀 법
clf.fit(X_train, np.argmax(Y_train, axis=1))

############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)



# 머신러닝의 정답률 구하기
ac_score5 = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score5)
print(pd.value_counts(pd.Series(y_test_estimated)))








import matplotlib.pyplot as plt
#ratio 설정
#times = [ac_score,ac_score2,ac_score3,ac_score4]
#timeslabels=["A","B","C","D"]
#plt.pie(times,labels=timeslabels,autopct="%.2f")
#autopct는 파이차트 안에 표시될 숫자의 형식을 지정하며 소수점 이하 두자리까지 표시하도록 함.
#plt.show()


data = [ac_score],[ac_score2],[ac_score3],[ac_score4],[ac_score5]
# 네 옵티마이저의 정확률을 박스플롯으로 비교
plt.boxplot(data,labels=["(1)","(2)","(3)","(4)","(5)"])
plt.title("Accuracy Comparison")
plt.grid()
plt.show()