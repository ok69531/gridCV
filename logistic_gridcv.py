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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def gridCV(model, data_x, data_y, param_dict):
    
    kf = KFold(n_splits = 5)
    params = ParameterGrid(param_dict)
    
    result_ = []
    metrics_ = ['parameters', 'macro_precision', 'weighted_precision', 
                'macro_recall', 'weighted_recall', 'macro_f1', 'weighted_f1', 'accuracy']
    
    for i in tqdm(range(len(params))):
        
        macro_precision_ = []
        weighted_precision_ = []
        macro_recall_ = []
        weighted_recall_ = []
        macro_f1_ = []
        weighted_f1_ = []
        accuracy_ =[]

        for train_idx, test_idx in kf.split(data_x):
            train_x, train_y = data_x.iloc[train_idx], data_y.iloc[train_idx]
            test_x, test_y = data_x.iloc[test_idx], data_y.iloc[test_idx]
            
            model_ = model(**params[i])
            model_.fit(train_x, train_y)
            pred = model_.predict(test_x)
            
            macro_precision_.append(precision_score(test_y, pred, average = 'macro'))
            weighted_precision_.append(precision_score(test_y, pred, average = 'weighted'))
            macro_recall_.append(recall_score(test_y, pred, average = 'macro'))
            weighted_recall_.append(recall_score(test_y, pred, average = 'weighted'))
            macro_f1_.append(f1_score(test_y, pred, average = 'macro'))
            weighted_f1_.append(f1_score(test_y, pred, average = 'weighted'))
            accuracy_.append(accuracy_score(test_y, pred))
    
        result_.append(dict(zip(metrics_, [params[i], 
                                           np.mean(macro_precision_),
                                           np.mean(weighted_precision_),
                                           np.mean(macro_recall_),
                                           np.mean(weighted_recall_),
                                           np.mean(macro_f1_),
                                           np.mean(weighted_f1_),
                                           np.mean(accuracy_)])))
    
    return pd.DataFrame(result_)


result = gridCV(LogisticRegression, train_mgl_x, train_mgl_y, logit_params)
result
