import pickle
import numpy as np
from glob import glob
from utils import voting

path_to_files = 'data/final_test'
path_to_model = 'models/model_test.pkl'
csv_name = 'results/output.csv'

print('\nObserving files')
files = list(sorted(glob(path_to_files + '/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0])))

print('\nLoading model')
with open('models/model_test.pkl', 'rb') as f:
    model = pickle.load(f)

print('\nReading files')
CLASSES = ['FNSZ', 'GNSZ', 'SPSZ', 'CPSZ', 'ABSZ', 'TNSZ', 'TCSZ']
with open(csv_name, 'w') as csvfile:
    for i, file_name in enumerate(files):
        print("{0}/{1}".format(i + 1, len(files)))
        d = pickle.load(open(file_name, 'rb'))
        if d.seizure_type == "MYSZ":
            continue
        x = d.data.astype('float32')
        x = np.vstack([np.reshape(e, 480) for e in x])

        y_pred = model.predict(x)
        print('real class:', CLASSES.index(d.seizure_type))
        label = voting(y_pred.astype(int))

        s = file_name[file_name.rfind('\\') + 1:]
        s = s[:s.find('.')]
        csvfile.write("{0} {1}".format(s, label))
        if i + 1 < len(files):
            csvfile.write("\n")






