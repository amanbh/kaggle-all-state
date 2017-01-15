import pandas as pd

model_keys = [
    'skl_Ridge-quad-100020161117T103454.json',
    'skl_Ridge-quad-100020161117T104911.json',
    'skl_Ridge-quad-100020161117T110254.json',
    'skl_Ridge-quad-100020161117T111651.json',
    'skl_Ridge-quad-100020161117T113232.json',
    'skl_Ridge-quad-100020161117T114551.json',
    'skl_Ridge-quad-100020161117T120118.json',
    'skl_Ridge-quad-100020161117T121525.json',
    'skl_Ridge-quad-100020161117T123053.json',
    'skl_Ridge-quad-100020161117T124430.json',
    ]
d = dict()
preds = np.zeros((x_train.shape[0],1))
for mk in model_keys:
    res = load_results_from_json(mk)
    preds += res['oof_train']
    d[mk] = mean_absolute_error(y_train, res['oof_train'])

d
mean_absolute_error(y_train, preds/len(d))

