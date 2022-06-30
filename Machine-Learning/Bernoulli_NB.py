# Bernoulli Naive Bayes
# 데이터 특징이 0 또는 1로 표현되었을 때 사용

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

email_list = [
                {'email title': 'free game only today', 'spam': True},
                {'email title': 'cheapest flight deal', 'spam': True},
                {'email title': 'limited time offer only today only today', 'spam': True},
                {'email title': 'today meeting schedule', 'spam': False},
                {'email title': 'your flight schedule attached', 'spam': False},
                {'email title': 'your credit card statement', 'spam': False}
             ]
df = pd.DataFrame(email_list)

df['label'] = df['spam'].map({True:1, False:0}) # 베르누이 특성 고려 true/false 를 1/0 으로 치환
df_x=df["email title"]
df_y=df["label"]
cv = CountVectorizer(binary=True)  # 특정 단어가 있으면 1, 없으명 0을 갖도록 설정
x_train=cv.fit_transform(df_x)
encoded_input=x_traincv.toarray()
cv.inverse_transform(encoded_input[0])
cv.get_feature_names()

# 베르누이 나이브 베이즈 모델 학습
bnb = BernoulliNB()
y_train=df_y.astype("int")
bnb.fit(x_traincv, y_train)

# 테스트 데이터 다듬기
test_email_list = [
                {'email title': 'free flight offer', 'spam': True},
                {'email title': 'hey traveler free flight deal', 'spam': True},
                {'email title': 'limited free game offer', 'spam': True},
                {'email title': 'today flight schedule', 'spam': False},
                {'email title': 'your credit card attached', 'spam': False},
                {'email title': 'free credit card offer only today', 'spam': False}
             ]
test_df = pd.DataFrame(test_email_list)
test_df['label'] = test_df['spam'].map({True:1,False:0})
test_x=test_df["email title"]
test_y=test_df["label"]
x_testcv=cv.transform(test_x)

predictions = bnb.predict(x_testcv)

accuracy_score(test_y, predictions)