import pickle
import random
import numpy as np
from models import VotingClassifier
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print('\nLoading data')
with open('data/p1_x_list.pkl', 'rb') as f:
    x = pickle.load(f)
y = np.load('data/p1_y_list.npy')
print(len(x))
print(len(y))

print('\nTransforming data')
x = [np.vstack([np.reshape(i, 480) for i in e]) for e in x]

print('\ntrain test split')
indexs = np.arange(0, len(y))
train_ids, test_ids, _, y_test = train_test_split(indexs, y, shuffle=True, random_state=42, test_size=0.05)
x_train = np.vstack([x[i] for i in train_ids])
x_test = [x[i] for i in test_ids]
y_train = np.hstack([[y[i]] * x[i].shape[0] for i in train_ids])
print(x_train.shape)
print(y_train.shape)
print(len(x_test))
print(y_test.shape)

print('\nTraining models')
model = VotingClassifier()
n_models = 100
# model_types = ['dt', 'rf', 'xgb', 'knn']
model_types = ['knn']
minsize = get_min_class_size(y_train)
for i in range(n_models):
    m_type = random.choice(model_types)
    print('Model {0}/{1}: {2}'.format(i + 1, n_models, m_type))
    m = new_model(30, m_type)

    ids = get_sample(minsize, y_train)
    x1 = x_train[ids, :]
    y1 = y_train[ids]

    m.fit(x1, y1)
    model.add_estimator(m)

print('\nTest')
y_pred = []
for i in range(len(x_test)):
    print('{0}/{1}'.format(i + 1, len(x_test)))
    prediction = model.predict(x_test[i]).astype(int)
    y_pred.append(voting(prediction))

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm = np.round(cm / np.sum(cm, axis=1, keepdims=True), 3)
print(acc)
print(cm)

with open('models/p1_model.pkl', 'wb') as f:
    pickle.dump(model, f)