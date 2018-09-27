import pickle
import matplotlib.pyplot as plt
import os

model_dir = 'models/shallow1'

with open(os.path.join(model_dir, 'eval_0_adv.pkl'), 'rb') as f:
    tr_adv = pickle.load(f)['threshold_results']

with open(os.path.join(model_dir, 'eval_0_adv_suppress_mask.pkl'), 'rb') as f:
    results = pickle.load(f)
    tr_adv_suppress = results['threshold_results']
    f1_adv_suppress = results['overall_result'].f1
with open(os.path.join(model_dir, 'eval_0_normal.pkl'), 'rb') as f:
    results = pickle.load(f)
    tr_normal = results['threshold_results']
    f1_normal = results['overall_result'].f1

print('F1 unperturbed: ', f1_normal)
print('F1 with perturbation: ', f1_adv_suppress)

rec_adv = []
prec_adv = []
for res in tr_adv:
    rec_adv.append(res.recall)
    prec_adv.append(res.precision)

rec_adv_suppress = []
prec_adv_suppress = []
for res in tr_adv_suppress:
    rec_adv_suppress.append(res.recall)
    prec_adv_suppress.append(res.precision)

rec_normal = []
prec_normal = []
for res in tr_normal:
    rec_normal.append(res.recall)
    prec_normal.append(res.precision)

plt.plot(rec_normal, prec_normal)
plt.plot(rec_adv, prec_adv)
plt.plot(rec_adv_suppress, prec_adv_suppress)
plt.show()