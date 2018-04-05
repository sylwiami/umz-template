#!/usr/bin/python3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie2/train')
report=pd.read_csv('train.tsv', sep='\t', names=['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMeters'])
reg=linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['rooms']), report['price'])
print(reg.coef_)
print(reg.predict(0))
os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie2/dev-0')
report2=pd.read_csv('in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrMeters'])
x_dev=pd.DataFrame(report2, columns=['rooms'])
y_dev_predict=reg.predict(x_dev)
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie2/test-A')
report3=pd.read_csv('in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location','sqrMeters']) 
x_test=pd.DataFrame(report3, columns=['rooms'])
y_test_predict=reg.predict(x_test)
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

sns.regplot(y=report["price"], x=report["rooms"]); plt.show()
