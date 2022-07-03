# 주성분분석

import pandas as pd
df = pd.DataFrame(columns = ['calory','breakfast','lunch','dinner','exercise','body_shape'])

df.loc[0] = [1200, 1, 0, 0, 2, 'Skinny']
df.loc[1] = [2800, 1, 1, 1, 1, 'Normal']
df.loc[2] = [3500, 2, 2, 1, 0, 'Fat']
df.loc[3] = [1400, 0, 1, 0, 3, 'Skinny']
df.loc[4] = [5000, 2, 2, 2, 0, 'Fat']
df.loc[5] = [1300, 0, 0, 1, 2, 'Skinny']
df.loc[6] = [3000, 1, 0, 1, 1, 'Normal']
df.loc[7] = [4000, 2, 2, 2, 0, 'Fat']
df.loc[8] = [2600, 0, 2, 0, 0, 'Normal']
df.loc[9] = [3000, 1, 2, 1, 1, 'Fat']

X = df[['calory','breakfast','lunch','dinner','exercise']]
X.head()

from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transfrom(X)

print(x_std)

Y = df[['body_shape']]
Y.head(10)

import numpy as np
features = x_std.T
covariance_marix = np.cov(features)
print(covariance_marix)

eig_vals, eig_vecs = np.linalg.eig(covariance_marix)
print('Eigenvectors \n%s' %eig_vecs)
print('Eigenvalues \n%s' %eig_vals)

eig_vals[0] / sum(eig_vals)

projected_X = x_std.dot(eig_vecs.T[0]) / np.linalg.norm(eig_vecs.T[0])
projected_X

result = pd.DataFrame(projected_X, columns=['PC1'])
result['y-axis'] = 0.0
reslut['label'] = Y

result.head(10)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.lmlplot(x='PC1', y='y=axis', data=result, fit_reg=False, scatter_kws={"s":50},hue="label")
plt.title('PCA result')

from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
sklearn_pca_x = pca.fit_transfrom(x_std)

sklearn_result = pd.DataFrame(sklearn_pca_x, columns=['PC1'])
sklearn_result['y_axis'] = 0.0
sklearn_result['label'] = Y
sns.lmplot('PC1', 'y-axis', data=sklearn_result, fit_reg=Fasle, scatter_kws={"s":50}, hue="label")