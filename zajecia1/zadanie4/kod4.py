#!/usr/bin/python3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie4/train')
report=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])
reg=linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['mileage', 'year']), report['price'])
x_train=pd.DataFrame(report, columns=['mileage', 'year'])
y_train_predict=reg.predict(x_train)
print(reg.coef_)
print(reg.intercept_)
pd.DataFrame(y_train_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie4/dev-0')
report2=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])
reg=linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['mileage', 'year']), report['price'])
x_dev=pd.DataFrame(report, columns=['mileage', 'year'])
y_dev_predict=reg.predict(x_dev)
print(reg.coef_)
print(reg.intercept_)
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

os.chdir('/home/students/s379489/Documents/umz-template/zajecia1/zadanie4/test-A')
report3=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity']) 
reg=linear_model.LinearRegression()
reg.fit(pd.DataFrame(report, columns=['mileage', 'year']), report['price'])
x_test=pd.DataFrame(report, columns=['mileage', 'year'])
y_test_predict=reg.predict(x_test)
print(reg.coef_)
print(reg.intercept_)
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

sns.regplot(y=report["price"], x=report["year"]); plt.show()
