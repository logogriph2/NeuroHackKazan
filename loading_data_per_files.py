from glob import glob
import pickle
import numpy as np

CLASSES = ['FNSZ', 'GNSZ', 'SPSZ', 'CPSZ', 'ABSZ', 'TNSZ', 'TCSZ']

def get_Xy(files):
    X, y = [], []

    for i, file_name in enumerate(files):
        print("{0}/{1}".format(i + 1, len(files)))
        d = pickle.load(open(file_name, 'rb'))
        if d.seizure_type == "MYSZ":
            X.append(0)
            y.append(-1)
        else:
            X.append(d.data.astype('float32'))
            y.append(CLASSES.index(d.seizure_type))

    return X, np.array(y)


files = list(sorted(glob('../data/pp_2/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0])))
print(len(files))

X, y = get_Xy(files)
np.save('data/p2_y_list_all', y)
with open('data/p2_x_list_all.pkl', 'wb') as f:
    pickle.dump(X, f)