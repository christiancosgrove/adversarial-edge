import pickle
import matplotlib.pyplot as plt
import os

model_dir = 'models/deep1'

with open(os.path.join(model_dir, 'eval_0_adv.pkl'), 'rb') as f:
    tr_adv = pickle.load(f)['threshold_results']
with open(os.path.join(model_dir, 'eval_0_normal.pkl'), 'rb') as f:
    tr_normal = pickle.load(f)['threshold_results']

rec_adv = []
prec_adv = []
for res in tr_adv:
    rec_adv.append(res.recall)
    prec_adv.append(res.precision)

rec_normal = []
prec_normal = []
for res in tr_normal:
    rec_normal.append(res.recall)
    prec_normal.append(res.precision)

plt.plot(rec_normal, prec_normal)
plt.plot(rec_adv, prec_adv)
plt.show()