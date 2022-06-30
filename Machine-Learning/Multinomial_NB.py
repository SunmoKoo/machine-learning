# Multinomial Naive Bayes 다항분포 나이브 베이즈
# 데이터 특징이 출현 횟수로 표현될때 사용

import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

review_list = [
                {'movie_review': 'this is great great movie. I will watch again', 'type': 'positive'},
                {'movie_review': 'I like this movie', 'type': 'positive'},
                {'movie_review': 'amazing movie in this year', 'type': 'positive'},
                {'movie_review': 'cool my boyfriend also said the movie is cool', 'type': 'positive'},
                {'movie_review': 'awesome of the awesome movie ever', 'type': 'positive'},
                {'movie_review': 'shame I wasted money and time', 'type': 'negative'},
                {'movie_review': 'regret on this move. I will never never what movie from this director', 'type': 'negative'},
                {'movie_review': 'I do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like actors in this movie', 'type': 'negative'},
                {'movie_review': 'boring boring sleeping movie', 'type': 'negative'}
             ]
df = pd.DataFrame(review_list)

df['label'] = df['type'].map({"positive":1, "negative":0})

df_x = df["movie_review"]
df_y = df["label"]

#다항분포 나이브베이즈의 입력 데이터는 고정된 크기의 벡터로써, 각각의 인덱스는 단어의 빈도수로 구분된 데이터이여야 합니다.  
#sklearn의 CountVectorizer를 사용하여 쉽게 구현할 수 있습니다.  
#CountVectorizer는 입력된 데이터(10개의 영화 리뷰)에 출현된 모든 단어의 갯수만큼의 크기의 벡터를 만든 후,  
#각각의 리뷰를 그 고정된 벡터로 표현합니다.  
cv = CountVectorizer()
x_traincv=cv.fit_transform(df_x)
encoded_input = x_traincv.toarray()
encoded_input

cv.inverse_transform(encoded_input[0])
cv.get_feature_names()

# 다항분포 나이브 베이즈 모델 학습
mnb = MultinomialNB()
y_train=df_y.astype('int')
mnb.fit(x_traincv, y_train)

test_feedback_list = [
                {'movie_review': 'great great great movie ever', 'type': 'positive'},
                {'movie_review': 'I like this amazing movie', 'type': 'positive'},
                {'movie_review': 'my boyfriend said great movie ever', 'type': 'positive'},
                {'movie_review': 'cool cool cool', 'type': 'positive'},
                {'movie_review': 'awesome boyfriend said cool movie ever', 'type': 'positive'},
                {'movie_review': 'shame shame shame', 'type': 'negative'},
                {'movie_review': 'awesome director shame movie boring movie', 'type': 'negative'},
                {'movie_review': 'do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like this boring movie', 'type': 'negative'},
                {'movie_review': 'aweful terrible boring movie', 'type': 'negative'}
             ]
test_df = pd.DataFrame(test_feedback_list)
test_df['label'] = test_df['type'].map({"positive":1,"negative":0})
test_x=test_df["movie_review"]
test_y=test_df["label"]

x_testcv=cv.transform(test_x)
predictions=mnb.predict(x_testcv)

accuracy_score(test_y, predictions)