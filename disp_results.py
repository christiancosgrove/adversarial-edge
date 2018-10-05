import pickle
import matplotlib.pyplot as plt
import os

model_dir = 'models/shallow1'

with open(os.path.join(model_dir, 'eval_0_adv_suppress.pkl'), 'rb') as f:
    results = pickle.load(f)
    tr_adv = results['threshold_results']
    f1_adv = results['overall_result'].f1

with open(os.path.join(model_dir, 'eval_0_adv_suppress_mask.pkl'), 'rb') as f:
    results = pickle.load(f)
    tr_adv_mask = results['threshold_results']
    f1_adv_mask = results['overall_result'].f1
with open(os.path.join(model_dir, 'eval_0_normal.pkl'), 'rb') as f:
    results = pickle.load(f)
    tr_normal = results['threshold_results']
    f1_normal = results['overall_result'].f1

print('F1 unperturbed: ', f1_normal)
print('F1 with perturbation: ', f1_adv)
print('F1 with perturbation (mask): ', f1_adv_mask)

rec_adv = []
prec_adv = []
for res in tr_adv:
    rec_adv.append(res.recall)
    prec_adv.append(res.precision)

rec_adv_mask = []
prec_adv_mask = []
for res in tr_adv_mask:
    rec_adv_mask.append(res.recall)
    prec_adv_mask.append(res.precision)

rec_normal = []
prec_normal = []
for res in tr_normal:
    rec_normal.append(res.recall)
    prec_normal.append(res.precision)


def precision_recall_chart(recall, precision):
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")


precision_recall_chart(rec_normal, prec_normal)
precision_recall_chart(rec_adv, prec_adv)
precision_recall_chart(rec_adv_mask, prec_adv_mask)
plt.legend(['normal', 'adversarial', 'mask'])
plt.show()