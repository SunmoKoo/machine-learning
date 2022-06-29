# Support Vector Machine 서포트 벡터 머신
# decision boundary(결정경계): 서로 다른 분류값을 결정하는 경계(n-1 dimension / hyperplane)
# support vector: 결정경계를 만드는 데 영향을 주는 최전방 데이터 포인트
# margin: 결정경계와 서포트 벡터 사이의 거리
# Linear SVM: 비용을 조절해서 마진의 크기를 조절할 수 있다.
# kernel trick: 선형분리가 불가능 할 경우 고차원으로 데이터를 옮김. 비용과 gamma를 조절해서 마진을 조절
# cost: 마진 너비 조절 변수. 클수록 마진 너비가 좁고, 작을수록 마진 넓이가 넓어진다.
# gamma: 커널의 표준편차 조절 변수, 작을수록 데이터포인트의 영향이 커져서 경계가 완만해지고, 클수록 경계가 구부러진다.

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

with open('../data/pkl/basketball_train.pkl', 'rb') as train_data:
    train = pickle.load(train_data)
    
with open('../data/pkl/basketball_test.pkl', 'rb') as test_data:
    test = pickle.load(test_data)


# sklearn의 gridsearch를 사용, 최적의 C, gamma를 구하기
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

def svc_param_selection(X, y, nfolds):
    svm_parameters = [{'kernel'}: ['rbf'], 'gamma': [0.00001, 0.0001, .0.001, 0.01, 0.1, 1],
                        'C': [0.01, 0.1, 1, 10, 100, 1000]]
clf = GridSearchCV(SVC(), svm_parameters, cv=10)
clf.fit(X_train, y_train.values.ravel())
print(clf.best_params_)

X_train = train[['3P', 'BLK']]
y_train = train[['Pos']]
clf = svc_param_selection(X_train, y_train.values.ravel(), 10)

# 결정 경계선 시각화
# 시각화를 하기 위해, 최적의 C와 최적의 C를 비교하기 위한 다른 C를 후보로 저장합니다.
C_canditates = []
C_canditates.append(clf.best_params_['C'] * 0.01)
C_canditates.append(clf.best_params_['C'])
C_canditates.append(clf.best_params_['C'] * 100)
# 시각화를 하기 위해, 최적의 gamma와 최적의 gamma를 비교하기 위한 다른 gamma를 후보로 저장합니다.
gamma_candidates = []
gamma_candidates.append(clf.best_params_['gamma'] * 0.01)
gamma_candidates.append(clf.best_params_['gamma'])
gamma_candidates.append(clf.best_params_['gamma'] * 100)

X = train[['3P', 'BLK']]
Y = train['Pos'].tolist()

# 포지션에 해당하는 문자열 SG와 C를 벡터화합니다.
position = []
for gt in Y:
    if gt == 'C':
        position.append(0)
    else:
        position.append(1)

# 각각의 파라미터에 해당하는 SVM 모델을 만들어 classifiers에 저장합니다.
classifiers = []
for C in C_canditates:
    for gamma in gamma_candidates:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X, Y)
        classifiers.append((C, gamma, clf))

# 18,18 사이즈의 챠트를 구성합니다.
plt.figure(figsize=(18, 18))
xx, yy = np.meshgrid(np.linspace(0, 4, 100), np.linspace(0, 4, 100))

# 각각의 모델들에 대한 결정 경계 함수를 적용하여 함께 시각화합니다.
for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 최적의 모델을 포함한 다른 파라미터로 학습된 모델들을 함께 시각화해봅니다.
    plt.subplot(len(C_canditates), len(gamma_candidates), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # 서포트 벡터와 결정경계선을 시각화합니다.
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X['3P'], X['BLK'], c=position, cmap=plt.cm.RdBu_r, edgecolors='k')

# 모델 테스트
# 테스트에 사용될 특징을 지정합니다
X_test = test[['3P', 'BLK']]
# 특징으로 예측할 값 (농구선수 포지션)을 지정합니다
y_test = test[['Pos']]
# 최적의 파라미터로 완성된 SVM에 테스트 데이터를 주입하여, 실제값과 예측값을 얻습니다.
y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))
print()
print("accuracy : "+ str(accuracy_score(y_true, y_pred)) )

# 실제값(ground truth)과 예측값(prediction)이 어느 정도 일치하는 눈으로 직접 비교해봅니다
comparison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_true.values.ravel()}) 
comparison