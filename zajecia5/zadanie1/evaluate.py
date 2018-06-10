from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import os
import pandas as pd
from keras import optimizers


r = pd.read_csv(os.path.join("train", "train.tsv"), header=None, names=[
                "Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"], sep='\t')
X_train = pd.DataFrame(
    r, columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
Y_train = pd.DataFrame(r, columns=["Occupancy"])


def create_baseline():
    # stworzenie modelu sieci neuronowej
    model = Sequential()
    # dodanie jednego neuronu, wejście do tego neuronu to ilość cech, funkcja aktywacji sigmoid, początkowe wartości wektorów to zero.
    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='zeros'))
    # stworzenie funkcji kosztu stochastic gradient descent
    sgd = optimizers.SGD(lr=0.1)
    # kompilacja modelu
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    # rysowanie architektury sieci, jeżeli ktoś ma zainstalowane odpowiednie biblioteki
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    return model


estimator = KerasClassifier(
    build_fn=create_baseline, epochs=3, verbose=True)


estimator.fit(X_train, Y_train)
predictions_train = estimator.predict(X_train)

# ACCURACY ON TRAINING DATA:
print('ACCURACY ON TRAINING DATA')
print((predictions_train == Y_train).mean())


r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=[
                "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"], sep='\t')
X_dev = pd.DataFrame(
    r, columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"),
                    header=None, names=["Occupation"], sep='\t')

predictions_dev = estimator.predict(X_dev)
print('ACCURACY ON DEV DATA')
print((predictions_dev == Y_dev).mean())

with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction[0]) + '\n')
