# Gaussian Naiave Bayes
# 특징들이 정규분포 가정 하 조건부 확률을 계산, 연속적인 성질이 있는 특징의 데이터 분류에 적합

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuray_score

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
df.target = df.target.map({0:"setosa", 1:"versicolor", 2:"virginica"})
df.head()

setosa_df = df[df.target == "setosa"]
versicolor_df = df[df.target == "versicolor"]
virginica_df = df[df.target == "virginica"]

ax = setosa_df['sepal length (cm)'].plot(kind='hist')
setosa_df['sepal length (cm)'].plot(kind-'kde', ax=ax, scondary_y=True, title="setosa sepal length (cm) distribution", figsize = (8,4))

ax = versicolor_df['sepal length (cm)'].plot(kind='hist')
versicolor_df['sepal length (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="versicolor sepal length", figsize = (8,4))

ax = virginica_df['sepal length (cm)'].plot(kind='hist')
virginica_df['sepal length (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="virginica sepal length", figsize = (8,4))

ax = setosa_df['sepal width (cm)'].plot(kind='hist')
setosa_df['sepal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="setosa sepal width", figsize = (8,4))

ax = versicolor_df['sepal width (cm)'].plot(kind='hist')
versicolor_df['sepal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="versicolor sepal width", figsize = (8,4))

ax = virginica_df['sepal width (cm)'].plot(kind='hist')
virginica_df['sepal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="virginica sepal width", figsize = (8,4))

ax = setosa_df['petal length (cm)'].plot(kind='hist')
setosa_df['petal length (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="setosa petal length", figsize = (8,4))

ax = versicolor_df['petal length (cm)'].plot(kind='hist')
versicolor_df['petal length (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="versicolor petal length", figsize = (8,4))

ax = virginica_df['petal length (cm)'].plot(kind='hist')
virginica_df['petal length (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="virginica petal length", figsize = (8,4))

ax = setosa_df['petal width (cm)'].plot(kind='hist')
setosa_df['petal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="setosa petal width", figsize = (8,4))

ax = versicolor_df['petal width (cm)'].plot(kind='hist')
versicolor_df['petal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="versicolor petal width", figsize = (8,4))

ax = virginica_df['petal width (cm)'].plot(kind='hist')
virginica_df['petal width (cm)'].plot(kind='kde', ax=ax, secondary_y=True, title="virginica petal width", figsize = (8,4))

# 데이터 다듬기
X_train, X_test, y_train, y_test = train_test_split(dataset.data, datset.target, test_size=0.2)

# 가우시안 나이브 베이즈 모델 학습
model = GaussianNB()
model.fit(X_train, y_train)

exepcted = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(y_test, predicted))

accuracy_score(y_test, predicted)