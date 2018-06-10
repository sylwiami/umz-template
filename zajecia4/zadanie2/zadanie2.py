import pandas as pd
import graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

file_name = 'answers-template'


# NO DEPTH RESTRAIN

clf = RandomForestClassifier()
clf = clf.fit(train_X, train_Y)

train_accuracy = sum(clf.predict(train_X) == train_Y) / len(train_X)
test_accuracy = sum(clf.predict(test_X) == test_Y) / len(test_X)

result_file = open(file_name, 'w')

result_file.write('TRAIN SET - NO DEPTH RESTRAIN\n')
result_file.write('accuracy: ' + str(train_accuracy) + '\n\n')
result_file.write('TEST SET - NO DEPTH RESTRAIN\n')
result_file.write('accuracy: ' + str(test_accuracy) + '\n\n')


# MAX DEPTH 4

clf = RandomForestClassifier(max_depth=4)
clf = clf.fit(train_X, train_Y)

train_accuracy = sum(clf.predict(train_X) == train_Y) / len(train_X)
test_accuracy = sum(clf.predict(test_X) == test_Y) / len(test_X)

result_file.write('TRAIN SET - MAX DEPTH 4\n')
result_file.write('accuracy: ' + str(train_accuracy) + '\n\n')
result_file.write('TEST SET - MAX DEPTH 4\n')
result_file.write('accuracy: ' + str(test_accuracy) + '\n\n')


# TEST WITH OWN PARAMETERS FROM ARRAYS

max_depths = [2, 4, 6, 8, 9]
min_samples_split = [0.2, 0.4, 0.6, 0.8, 0.9]
min_samples_leaf = [2, 4, 6, 8, 9]
bootstrap = [True, False]

results = []

for i in max_depths: 
    for j in min_samples_split:       
        for k in min_samples_leaf:
            for m in bootstrap:
                clf = RandomForestClassifier(max_depth=i, min_samples_split=j, min_samples_leaf=k, bootstrap=m)
                clf = clf.fit(train_X, train_Y)

                train_accuracy = sum(clf.predict(train_X) == train_Y) / len(train_X)
                test_accuracy = sum(clf.predict(test_X) == test_Y) / len(test_X)
                element =  (i, j, k, m, train_accuracy, test_accuracy)
                results.append(element)


# SORT RESULTS AND GET BEST 10

results_sorted = sorted(results, key=lambda element: (element[5], element[4]), reverse=True)

result_file.write('YOUR OWN FOREST PARAMETERS:\n\n')
result_file.write('BEST 10 RESULTS\n\n')

for i in results_sorted[:10]:
    result_file.write('TRAIN SET - MAX DEPTH ' + str(i[0]) + ' MIN_SAMPLES_SPLIT ' + str(i[1]) + ' MIN_SAMPLES_LEAF ' + str(i[2]) + ' BOOTSTRAP ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[4]) + '\n\n')

    result_file.write('TEST SET - MAX DEPTH ' + str(i[0]) + ' MIN_SAMPLES_SPLIT ' + str(i[1]) + ' MIN_SAMPLES_LEAF ' + str(i[2]) + ' BOOTSTRAP ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[5]) + '\n\n')

result_file.close()
