#%%
import time
import random
import openpyxl
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import column 

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import scipy.stats as stats

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#%%
'''
    read data
'''

path = 'C:/Users/SOYOUNG/Desktop/toxic/data/oecd_echemportal/Preprocessed data/lc50_split/'

mgl = pd.read_excel(path + 'train_mgl.xlsx', sheet_name = 'Sheet1')
ppm = pd.read_excel(path + 'train_ppm.xlsx', sheet_name = 'Sheet1')


def Smiles2Fing(smiles):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
    maccs_bit = [i.ToBitString() for i in maccs]
    
    fingerprints = pd.DataFrame({'maccs': maccs_bit})
    fingerprints = fingerprints['maccs'].str.split(pat = '', n = 167, expand = True)
    fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = ['maccs_' + str(i) for i in range(1, 168)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)
    
    return ms_none_idx, fingerprints


mgl_drop_idx, mgl_fingerprints = Smiles2Fing(mgl.SMILES)
mgl_y = mgl.value.drop(mgl_drop_idx).reset_index(drop = True)
mgl_y = pd.DataFrame({'value': mgl_y,
                     'category': pd.cut(mgl_y, bins = [0, 0.5, 2.0, 10, 20, np.infty], labels = range(5))})
# mgl_y = pd.DataFrame({'value': mgl_y,
#                      'category': pd.qcut(mgl_y, 5, labels = range(5))})
mgl_y['category'].value_counts().sort_index()
mgl_y['category'].value_counts(normalize = True).sort_index()


ppm_drop_idx, ppm_fingerprints = Smiles2Fing(ppm.SMILES)
ppm_y = ppm.value.drop(ppm_drop_idx).reset_index(drop = True)
ppm_y = pd.DataFrame({'value': ppm_y,
                      'category': pd.cut(ppm_y, bins = [0, 100, 500, 2500, 20000, np.infty], labels = range(5))})
# ppm_y = pd.DataFrame({'value': ppm_y,
#                       'category': pd.qcut(ppm_y, 5, labels = range(5))})


#%%
from itertools import product
from collections.abc import Iterable


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid


# for example
logit_params = {'random_state': [0], 
                'C': [0.1, 1, 10], 
                'max_iter': [50, 100, 150]}

params = ParameterGrid(logit_params)


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold


kf = KFold(n_splits = 5)

result_ = []
metrics = ['macro_precision', 'weighted_precision', 'macro_recall', 
           'weighted_recall', 'macro_f1', 'weighted_f1', 'accuracy']
train_metrics = list(map(lambda x: 'train_' + x, metrics))
val_metrics = list(map(lambda x: 'val_' + x, metrics))


for i in tqdm(range(len(params))):
    
    train_macro_precision_, train_weighted_precision_ = [], []
    train_macro_recall_, train_weighted_recall_ = [], []
    train_macro_f1_, train_weighted_f1_ = [], []
    train_accuracy_ =[]

    val_macro_precision_, val_weighted_precision_ = [], []
    val_macro_recall_, val_weighted_recall_ = [], []
    val_macro_f1_, val_weighted_f1_ = [], []
    val_accuracy_ =[]

    for train_idx, val_idx in kf.split(mgl_fingerprints):
        train_x, train_y = mgl_fingerprints.iloc[train_idx], mgl_y.category.iloc[train_idx]
        val_x, val_y = mgl_fingerprints.iloc[val_idx], mgl_y.category.iloc[val_idx]
        
        model = LogisticRegression(**params[i])
        model.fit(train_x, train_y)
        
        train_pred = model.predict(train_x)
        val_pred = model.predict(val_x)
        
        train_macro_precision_.append(precision_score(train_y, train_pred, average = 'macro'))
        train_weighted_precision_.append(precision_score(train_y, train_pred, average = 'weighted'))
        train_macro_recall_.append(recall_score(train_y, train_pred, average = 'macro'))
        train_weighted_recall_.append(recall_score(train_y, train_pred, average = 'weighted'))
        train_macro_f1_.append(f1_score(train_y, train_pred, average = 'macro'))
        train_weighted_f1_.append(f1_score(train_y, train_pred, average = 'weighted'))
        train_accuracy_.append(accuracy_score(train_y, train_pred))

        val_macro_precision_.append(precision_score(val_y, val_pred, average = 'macro'))
        val_weighted_precision_.append(precision_score(val_y, val_pred, average = 'weighted'))
        val_macro_recall_.append(recall_score(val_y, val_pred, average = 'macro'))
        val_weighted_recall_.append(recall_score(val_y, val_pred, average = 'weighted'))
        val_macro_f1_.append(f1_score(val_y, val_pred, average = 'macro'))
        val_weighted_f1_.append(f1_score(val_y, val_pred, average = 'weighted'))
        val_accuracy_.append(accuracy_score(val_y, val_pred))

    result_.append(dict(zip(list(params[i].keys())+ train_metrics + val_metrics, 
                            list(params[i].values()) + 
                            [np.mean(train_macro_precision_), np.mean(train_weighted_precision_),
                             np.mean(train_macro_recall_), np.mean(train_weighted_recall_),
                             np.mean(train_macro_f1_), np.mean(train_weighted_f1_),
                             np.mean(train_accuracy_),
                             np.mean(val_macro_precision_), np.mean(val_weighted_precision_),
                             np.mean(val_macro_recall_), np.mean(val_weighted_recall_),
                             np.mean(val_macro_f1_), np.mean(val_weighted_f1_),
                             np.mean(val_accuracy_)])))


result = pd.DataFrame(result_)
result


#%%
from matplotlib import pyplot as plt


logit_f1 = result.groupby(['C'])[['train_weighted_f1', 'val_weighted_f1']].mean().reset_index()


plt.plot(logit_f1.C, logit_f1.train_weighted_f1)
plt.plot(logit_f1.C, logit_f1.val_weighted_f1)
plt.title('F1 score')
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()