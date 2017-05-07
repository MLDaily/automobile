# dataset

import pandas as pd
import numpy as np
from datetime import datetime, date, time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

cols = ['symboling', 'normalised-losses', 'make', 'fuel-type',
        'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
        'engine-location', 'wheel-base', 'length', 'width', 'height',
        'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size',
        'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

file = pd.read_csv('data/automobile.csv', names=cols)
file = file[file['price'] != '?']
file = file.apply(lambda x: x.fillna(method='pad'), axis=0)
# train = file[cols[:-1]]
fh = FeatureHasher(n_features=25, input_type="string")
# Xcat = fh.transform(np.asarray(train.astype(str)))


# for i, chunk in enumerate(file):
#     chunk = chunk.apply(lambda x: x.fillna(method='pad'), axis=0)
#     if i % 2 == 0:
#         chunk.to_csv('data/train.csv', header=False, index=False, mode='a')
#     else:
#         chunk.to_csv('data/test.csv', header=False, index=False, mode='a')

# Train classifier
# clf = RandomForestRegressor()
# clf = LogisticRegression()
clf = LinearRegression()
# clf = KNeighborsRegressor()
# clf = MLPRegressor()
# clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
all_classes = np.array([0, 1])

y_train = file["price"]
train = file[cols[:-1]]
# train.drop(["normalised-losses"], axis=1, inplace=True)
Xcat = fh.transform(np.asarray(train.astype(str)))
print 'Training'
clf.fit(Xcat, y_train)

test_file = pd.read_csv('data/test.csv', names=cols)
# test_file = test_file.apply(lambda x: x.fillna(method='pad'), axis=0)

test = test_file[cols[:-1]]
y_test = test_file["price"]
# test.drop(["normalised-losses"], axis=1, inplace=True)
X_test = fh.transform(np.asarray(test.astype(str)))

y_pred = clf.predict(X_test)

for i, value in enumerate(y_test):
    print value, y_pred[i]

print clf.residues_
# print clf.densify()
# print clf.decision_function(X_test)
# import matplotlib.pyplot as plt

# flt = ['normalised-losses', 'wheel-base', 'length', 'width', 'height', 'bore',
#        'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'price']
# inttp = ['symboling', 'curb-weight', 'engine-size', 'city-mpg',
# 'highway-mpg']

# print list(file.columns[file.dtypes == np.int64])

# for col in cols:
#     f = file[[col, 'price']]
#     if col not in flt and col not in inttp:
#         f[col].value_counts().plot(x=col, y='price', kind='bar', alpha=1)
#     else:
#         f.plot.hist(y=col, alpha=0.2, bins=50)
#     # f.show()
#     plt.savefig('images/' + col + '.png')
#     plt.close()

# ls = ['peak-rpm', 'engine-size', 'curb-weight', 'price']
# f = file[ls].cumsum()

# f.plot.hist(x='price', alpha=1)
# plt.savefig('images/' + '-'.join(ls[:-1]) + '.png')
# plt.close()
