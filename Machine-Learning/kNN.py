# jupyter notebook으로 연습
# k-Nearest Neighbor 최근접 이웃
# 지도학습(Supervised Learning): training data + label
# 데이터 분류(Classification): data -> class label(x: attribute set, y: target attribute)
# practice: basketball data

import pandas as pd

# read data
df = pd.read_csv('Machine-Learning\basketball_stat.csv')
# check data
df.head()
df.Pos.value_counts()

# 시각화 출력
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline  # 플로팅 명령의 출력이 Jupyter Notebook과 같은 프론트에서 실행하면 결과를 셀 아래 inline으로 표시

sns.lmplot(x='STL', y='2P', data=df, fit_reg=False, scatter_kws={"s": 150}, markers=["o", "x"], hue="Pos")
#          x축,     y축,    데이터,  라인 없음,      점의 크기,              점 모양,            예측값
plt.title('STL and 2P in 2d plane')  # title / 스틸과 2점슛은 불필요 데이터임을 판단

sns.lmplot(x='BLK', y='3P', data=df, fit_reg=False, scatter_kws={"s": 150}, markers=["o", "x"], hue="Pos")
plt.title('BLK and 3P in 2d plane')  # 블로킹과 3점슛은 필요 데이터임을 판단

# 데이터 다듬기
df.drop(['2P', 'AST', 'STL'], axis=1, inplace = True)
# inplace=False이면 원본 DataFrame은 직접적으로 drop 변경하지 않으며 원본에서 데이터가 drop 된 새로운 DataFrame을 반환
df.head()

# 데이터 나누기(학습할 데이터와 테스트 테이터)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)  # 20%를 테스트 데이터로 분류
train.shape[0]
test.shape[0]  # 테스트 데이터 개수 확인

# optimal k parameter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score  # cross_val_score: k-fold 교차검증 수행

max_k_range = train.shape[0] // 2   # k-fold k 수행 범위는 3 ~ 학습데이터 크기의 절반
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)

cross_validation_scores = []
x_train = train[['3P', 'BLK', 'TRB']]
y_train = train[['Pos']]

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())

cross_validation_scores

# 시각화
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show

# select optimal k
k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print("The best number of k : " + str(k))

# model test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

x_train = train[['3P', 'BLK', 'TRB']]
y_train = train[['Pos']]

knn.fit(x_train, y_train.values.ravel())
x_test = test[['3P', 'BLK', 'TRB']]
y_train = test[['Pos']]

pred = knn.predict(x_test)
#print("accuracy : " + str(accuracy_score(y_test.values.ravel(),pred)))

comparison = pd.DataFrame({'prediction':pred, 'ground_truth':y_test.values.ravel()})
comparison