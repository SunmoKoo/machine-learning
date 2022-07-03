# 군집화
import pandas as pd 
import unmpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 데이터 조성
df = pd.DataFrame(columns=['height','weight'])
df.loc[0] = [185,60]
df.loc[1] = [180,60]
df.loc[2] = [185,70]
df.loc[3] = [165,63]
df.loc[4] = [155,68]
df.loc[5] = [170,75]
df.loc[6] = [175,80]

# 데이터 시각화
sns.lmplot(x='height', y='weight', data=df, fit_reg=False, scatter_kws={"s":200})

# k-means cluster
data_points = df.values
kmeans = KMeans(n_clusters=3).fit(data_points)

# 시각화
sns.lmplot('height', 'weight', data=df, fit_reg=False, scatter_kws={"s":150}, hue="cluster_id")