##### 화자인식 일반 머신러닝 코드 #####
from turtle import update

import folder as folder
import librosa
import librosa.display
import pyaudio  # 마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, svm
from sklearn.externals import joblib
import os
import warnings
import speech_recognition as sr

from sklearn.svm._libsvm import predict

warnings.simplefilter("ignore", DeprecationWarning)

##### 변수 설정 부분 #####
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 비트레이트 설정
CHUNK = int(RATE / 10)  # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 1  # 녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
DATA_PATH = "C:/Users/JISU/OneDrive/바탕 화면/data"
#C:\Users\JISU\OneDrive\바탕 화면\data
#DATA_PATH = "./OneDrive/바탕 화면/프종/프종/data"
train_data = []  # train_date 저장할 공강
train_label = []  # train_label 저장할
test_data = []  # train_date 저장할 공강
test_label = []  # train_label 저장할
################################맨처음##################################################
############################### STT부분 - 그룹 골라오기 코드 ################################

# microphone에서 auido source를 생성합니다
#r = sr.Recognizer()
#with sr.Microphone() as source:
    #print("SPEAK YOUR NAME!")
    #audio = r.listen(source)

# 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
#try:
    #print("Google Speech Recognition thinks you said : " + r.recognize_google(audio, language='ko'))
#except sr.UnknownValueError:
    #print("Google Speech Recognition could not understand audio")
#except sr.RequestError as e:
    #print("Could not request results from Google Speech Recognition service; {0}".format(e))

# 결과
# Google Speech Recognition thinks you said : 안녕하세요

############## wav 파일로 저장###################################
# write audio to a WAV file
#with open("microphone-results.wav", "wb") as f:
    #f.write(audio.get_wav_data())


#############################3##########GUI_KIVY_PART####################################
###################################

def load_wave_generator(path):
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    folders = os.listdir(path)
    # while True:
    # print("loaded batch of %d files" % len(files))
    for folder in folders:
        if not os.path.isdir(path): continue  # 폴더가 아니면 continue
        files = os.listdir(path + "/" + folder)
        print("Foldername :", folder, "-", len(files))  # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            else:
                global train_data, train_label  # 전역변수를 사용하겠다.
                print("Filename :", wav)  # .wav 파일이 아니면 continue
                y, sr = librosa.load(path + "/" + folder + "/" + wav)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=35, hop_length=int(sr * 0.8), n_fft=int(sr * 1)).T
                if (len(train_data) == 0):
                    train_data = mfcc
                    train_label = np.full(len(mfcc), str(folder))
                else:
                    train_data = np.concatenate((train_data, mfcc), axis=0)
                    train_label = np.concatenate((train_label, np.full(len(mfcc), str(folder))), axis=0)
                    # print("mfcc :",mfcc.shape)
    #if (r.recognize_google(audio, language='ko') == str(folder)):
               #print(str(folder))

load_wave_generator(DATA_PATH)







######## 음성 데이터를 녹음 해 저장하는 부분 ########
p = pyaudio.PyAudio()  # 오디오 객체 생성

stream = p.open(format=FORMAT,  # 16비트 포맷
                channels=CHANNELS,  # 모노로 마이크 열기
                rate=RATE,  # 비트레이트
                input=True,
                frames_per_buffer=CHUNK)  # CHUNK만큼 버퍼가 쌓인다.

print("Start to record the audio.")


frames = []  # 음성 데이터를 채우는 공간

for i in range(0, int(RATE / CHUNK * 10)):
    # 지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording is finished.")

stream.stop_stream()  # 스트림닫기
stream.close()  # 스트림 종료
p.terminate()  # 오디오객체 종료

# WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

spf = wave.open(WAVE_OUTPUT_FILENAME, 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, dtype=np.int16)

# 시간 흐름에 따른 그래프를 그리기 위한 부분
Time = np.linspace(0, len(signal) / RATE, num=len(signal))

plt.figure(1)
plt.title('Voice Signal Wave...')
#plt.plot(signal) // 음성 데이터의 그래프
plt.plot(Time, signal)
plt.show()


######## 음성 데이터를 읽어와 학습 시키는 부분 ########

print("train_data.shape :", train_data.shape, type(train_data))
print("train_label.shape :", train_label.shape, type(train_label))
# print(mfcc[0])
# print(train_label)
clf = LogisticRegression()
clf.fit(train_data, train_label)


y, sr = librosa.load("./output.wav")
#y, sr = librosa.load("./test_홍지수.wav")
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr)
# y, sr = librosa.load(WAVE_OUTPUT_FILENAME)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T

y_test_estimated = clf.predict(mfcc)
print(y_test_estimated)
test_label = np.full(len(mfcc), 0)
print(test_label)
'''
0 유인나
1 배철수
2 이재은
3 최일구
4 문재인 대통령
5 김하은
6 이다지
7 이국종
8 동빈나
9 허민석
10 주언규
11 서장훈
12 정승제
13 전한길
14 설민석
15 홍지수
16 태연
17 웬디
18 아이유
19 태민
20 조정석
'''

#ac_score = metrics.accuracy_score(y_test_estimated, test_label)
# 정답률 구하기
#print("정답률 =",metrics.accuracy_score(y_test_estimated, test_label) )
print(pd.value_counts(pd.Series(y_test_estimated)))

#load_wave_generator(DATA_PATH)
#voices = datasets.load_files()
#clf = LogisticRegression()
#clf.fit(train_data, train_label)
#joblib.dump(clf,'voices.pkl',compress=True)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
