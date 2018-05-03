import pandas as pd

import seaborn as sns

import os

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


rtrain = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', header=None)

rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', header=None)

rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'], header=None)

rprod = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', header=None)


lr_full = LogisticRegression()
	
X = pd.DataFrame(rtrain.loc[:, 1:])
lr_full.fit(X, rtrain[0])

occupancy_v = str((sum(rtrain[0] == 'g') / len(rtrain)))

zero_rule_v = str(1 - sum(rtrain[0] == 'g') / len(rtrain))


tp_v = sum((lr_full.predict(X) == rtrain[0]) & (lr_full.predict(X) == 'g'))

tn_v = sum((lr_full.predict(X) == rtrain[0]) & (lr_full.predict(X) == 'b'))

fp_v = sum((lr_full.predict(X) != rtrain[0]) & (lr_full.predict(X) == 'g'))

fn_v = sum((lr_full.predict(X) != rtrain[0]) & (lr_full.predict(X) == 'b'))


accuracy_v = str((tp_v + tn_v) / len(rtrain))

sensivity_v = str(tp_v / (tp_v + fn_v))

specifity_v = str(tn_v / (fp_v + tn_v))


dev_occupancy_v = str(sum(rdev_expected['y'] == 'g') / len(rdev_expected))

dev_zero_rule_v = str(1 - sum(rdev_expected['y'] == 'g') / len(rdev))


dev_tp_v = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 'g'))

dev_tn_v = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 'b'))

dev_fp_v = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 'g'))

dev_fn_v = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 'b'))


dev_accuracy_v = str((dev_tp_v + dev_tn_v) / len(rdev))

dev_sensivity_v = str(dev_tp_v / (dev_tp_v + dev_fn_v))

dev_specifity_v = str(dev_tn_v / (dev_fp_v + dev_tn_v))

#Zapisanie do pliku notes

output = open('notes.txt', 'w')

output.write('dane treningowe\n')

output.write('=' * 100 + '\n')

output.write('Occupancy: ' + occupancy_v +'%\n')

output.write('Dokładność modelu zerowego na danych treningowych: ' + zero_rule_v +'\n') 

output.write('=' * 100 + '\n')

output.write('accuracy: ' + accuracy_v +'\n')

output.write('sensivity: ' + sensivity_v +'\n')

output.write('specifity: ' + specifity_v +'\n')

output.write('=' * 100 + '\n')

output.write('Macierz\n')

output.write('True Positives: ' + str(tp_v) +'\n')

output.write('True Negatives: ' + str(tn_v) + '\n')

output.write('False Positives: ' + str(fp_v) + '\n')

output.write('False Negatives: ' + str(fn_v) + '\n')

output.write('=' * 100 + '\n\n\n\n')

output.write('dane developerskie\n')

output.write('=' * 100 + '\n')

output.write('Occupancy: ' + dev_occupancy_v +'%\n')

output.write('Dokładność modelu zerowego na danych developerskich: ' + dev_zero_rule_v +'\n') 

output.write('=' * 100 + '\n')

output.write('accuracy: ' + dev_accuracy_v +'\n')

output.write('sensivity: ' + dev_sensivity_v +'\n')

output.write('specifity: ' + dev_specifity_v +'\n')

output.write('=' * 100 + '\n')

output.write('Macierz\n')

output.write('True Positives: ' + str(dev_tp_v) +'\n')

output.write('True Negatives: ' + str(dev_tn_v) + '\n')

output.write('False Positives: ' + str(dev_fp_v) + '\n')

output.write('False Negatives: ' + str(dev_fn_v) + '\n')

output.write('=' * 100 + '\n\n')

output.close()


file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr_full.predict(rdev)):

	file.write(str(line)+'\n')

file.close()


file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr_full.predict(rprod)):

	file.write(str(line)+'\n')

file.close()



