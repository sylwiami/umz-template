import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']


def save_plot():
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.columns[:-1],
                                    class_names=[str(x)
                                                 for x in [1, 2, 3, 4, 5, 6, 7]],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render('my-tree')

max_depths = [1, 2, 3, 5, 7]
min_samples_split = [0.1, 0.2, 0.3, 0.5, 0.7]
min_samples_leaf = [1, 2, 3, 5, 7]
presort = [True, False]

results = []

# TEST WITH OWN PARAMETERS FROM ARRAYS

for i in max_depths: 
    for j in min_samples_split:       
        for k in min_samples_leaf:
            for m in presort:
                clf = tree.DecisionTreeClassifier(max_depth=i, min_samples_split=j, min_samples_leaf=k, presort=m)
                clf = clf.fit(train_X, train_Y)

                train_accuracy = sum(clf.predict(train_X) == train_Y) / len(train_X)
                test_accuracy = sum(clf.predict(test_X) == test_Y) / len(test_X)
                element =  (i, j, k, m, train_accuracy, test_accuracy)
                results.append(element)


# SORT RESULTS AND GET BEST 10

results_sorted = sorted(results, key=lambda element: (element[5], element[4]), reverse=True)

result_file = open('parameters', 'w')
result_file.write('BEST 10 RESULTS\n\n')

for i in results_sorted[:10]:

    result_file.write('TRAIN SET - MAX DEPTH ' + str(i[0]) + ' MIN_SAMPLES_SPLIT ' + str(i[1]) + ' MIN_SAMPLES_LEAF ' + str(i[2]) + ' PRESORT ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[4]) + '\n\n')

    result_file.write('TEST SET - MAX DEPTH ' + str(i[0]) + ' MIN_SAMPLES_SPLIT ' + str(i[1]) + ' MIN_SAMPLES_LEAF ' + str(i[2]) + ' PRESORT ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[5]) + '\n\n')

result_file.close()

# SAVE PLOT FOR BEST RESULT
clf = tree.DecisionTreeClassifier(max_depth=results_sorted[0][0], min_samples_split=results_sorted[0][1], min_samples_leaf=results_sorted[0][2], presort=results_sorted[0][3])
clf = clf.fit(test_X, test_Y)
save_plot()
