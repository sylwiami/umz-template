import pandas as pd
import graphviz
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

file_name = 'answers-template'


# TEST WITH OWN PARAMETERS FROM ARRAYS

number_of_neighbors = [1, 2, 3, 4, 8, 15, 25, 26, 29]
algorithms = ['ball_tree', 'kd_tree', 'brute']
weights = ['uniform', 'distance']
leaf_sizes = [1, 2, 3, 4, 8, 13, 20, 35]
results = []

result_file = open(file_name, 'w')

for i in number_of_neighbors:
    for j in algorithms:
        for k in weights:
            for l in leaf_sizes:
                clf = KNeighborsClassifier(n_neighbors=i, algorithm=j, weights=k, leaf_size=l)
                clf = clf.fit(train_X, train_Y)

                train_accuracy = sum(clf.predict(train_X) == train_Y) / len(train_X)
                test_accuracy = sum(clf.predict(test_X) == test_Y) / len(test_X)
                element =  (i, j, k, l, train_accuracy, test_accuracy)
                results.append(element)


# SORT RESULTS AND GET BEST 10

results_sorted = sorted(results, key=lambda element: (element[5], element[4]), reverse=True)

result_file.write('YOUR OWN NUMBER OF NEIGHBORS:\n\n')
result_file.write('BEST 10 RESULTS\n\n')

for i in results_sorted[:10]:
    result_file.write('TRAIN SET - NUMBER OF NEIGHBORS: ' + str(i[0]) + ' ALGORITHM: ' + str(i[1]) + ' WEIGHTS: ' + str(i[2]) + ' LEAF SIZE: ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[4]) + '\n\n')

    result_file.write('TEST SET - NUMBER OF NEIGHBORS: ' + str(i[0]) + ' ALGORITHM: ' + str(i[1]) + ' WEIGHTS: ' + str(i[2]) + ' LEAF SIZE: ' + str(i[3]) + '\n')
    result_file.write('accuracy: ' + str(i[5]) + '\n\n')
    result_file.write(20 * '-' + '\n\n')

result_file.close()
