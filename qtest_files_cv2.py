import pickle
import os
import random
import numpy as np
from models import VotingClassifier
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score



print('\nAll files names')
all_files = os.listdir('../data/pp_1')

print('\nLoading kfold file')
kfold = []
with open('data/cv-5.txt') as sf:
    data = sf.read()
    for i in range(1, 6):
        setlist = data.split('[')[i].split(']')[0].split(' ')
        setlist = [e.replace("'", "").strip() for e in setlist]
        ids = [all_files.index(e) for e in setlist]
        kfold.append(ids[:len(ids)//8])

print('\nLoading data')
with open('data/p1_x_list_all.pkl', 'rb') as f:
    x = pickle.load(f)
y = np.load('data/p1_y_list_all.npy')
print(len(x))
print(len(y))

print('\nTransforming data')
x = [np.vstack([np.reshape(i, 480) for i in e]) if isinstance(e, np.ndarray) else e for e in x]

print('\nCross validation')
n_models = 10
# model_types = ['dt', 'rf', 'xgb', 'knn']
model_types = ['dt', 'rf', 'knn']
scores = []
for cv_index in range(5):
    print('===== KFold {0}/5 ======'.format(cv_index + 1))
    test_ids = kfold[cv_index]
    train_ids = [i for i in range(len(y)) if (y[i] >= 0) and (i not in test_ids)]
    x_train = np.vstack([x[i] for i in train_ids])
    y_train = np.hstack([[y[i]] * x[i].shape[0] for i in train_ids])
    x_test = [x[i] for i in test_ids]
    y_test = [y[i] for i in test_ids]
    print('x train shape', x_train.shape)

    print('\nTraining models')
    model = VotingClassifier()
    minsize = get_min_class_size(y_train)
    for i in range(n_models):
        m_type = random.choice(model_types)
        print('Model {0}/{1}: {2}'.format(i + 1, n_models, m_type))
        m = new_model(25, m_type)

        ids = get_sample(minsize, y_train)
        x1 = x_train[ids, :]
        y1 = y_train[ids]

        m.fit(x1, y1)
        model.add_estimator(m)

    print('\nTest')
    y_pred = []
    y_true = []
    for i in range(len(x_test)):
        if not isinstance(x_test[i], np.ndarray):
            print('SKIPPED')
            continue
        print('{0}/{1}'.format(i + 1, len(x_test)))
        prediction = model.predict(x_test[i]).astype(int)
        y_pred.append(voting(prediction))
        y_true.append(y_test[i])
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = np.round(cm / np.sum(cm, axis=1, keepdims=True), 3)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f1)
    print(acc)
    print(cm)
    scores.append(f1)
    with open('model_{}.pkl'.format(int(f1*100)), 'wb') as mf:
        pickle.dump(model, mf)

print('==========================')
print(scores)
print(np.mean(scores))





