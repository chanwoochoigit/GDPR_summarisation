import argparse

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import list_2d_to_nparray, get_npy_path, get_bin_path, get_bow_model_path, encode_binary_labels
from joblib import dump

class SensitivitySpecificityCallback(Callback):

    def __init__(self, validation_data):
        super(SensitivitySpecificityCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_test = self.validation_data[0]
        y_test = self.validation_data[1]
        # x_test, y_test = self.validation_data
        predictions = self.model.predict(x_test)
        y_test = np.argmax(y_test, axis=-1)
        predictions = np.argmax(predictions, axis=-1)
        c = confusion_matrix(y_test, predictions)

        print('Confusion matrix:\n', c)
        print('sensitivity', c[0, 0] / (c[0, 1] + c[0, 0]))
        print('specificity', c[1, 1] / (c[1, 1] + c[1, 0]))

def to_sqaured(label_1d):
    squared_labels = []
    for lb in label_1d:
        if lb == 0:
            squared_labels.append([0,1])
        elif lb == 1:
            squared_labels.append([1,0])
        else:
            sys.exit("Wrong input in the supposed label array!")

    return squared_labels

def go_baseline_training(X_train, X_test, y_train, y_test, callback):
    model = Sequential()

    n_input_neurons = X_train.shape[1]
    n_output_neurons = 2
    n_hidden_neurons = round((X_train.shape[1] + n_output_neurons) * 2 / 3)

    model.add(Dense(n_input_neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(n_hidden_neurons, activation='relu'))
    model.add(Dense(n_hidden_neurons, activation='relu'))
    model.add(Dense(n_output_neurons, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=callback)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def go_training(X_train, X_test, y_train, y_test, callback):
    model = Sequential()

    n_input_neurons = X_train.shape[1]
    n_output_neurons = 2
    n_hidden_neurons = round((X_train.shape[1] + n_output_neurons) * 2 / 3)

    """ input layer """
    model.add(Dense(n_input_neurons, input_dim=X_train.shape[1], kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    """ hidden layer 1 """
    model.add(Dense(n_hidden_neurons, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    """ hidden layer 2 """
    model.add(Dense(n_input_neurons, input_dim=X_train.shape[1], kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    """ hidden layer 3 """
    model.add(Dense(n_hidden_neurons, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # """ hidden layer 4 """
    # model.add(Dense(n_hidden_neurons, kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))

    """output layer with softmax activation"""
    model.add(Dense(n_output_neurons, activation='sigmoid'))

    """ Compile model and print summary"""
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test), callbacks=callback)

    """ plot training and test result """
    plt.plot(history.history['accuracy'], label='MAE (training data)')
    plt.title('Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

if __name__ == '__main__':

    # take flags
    parser = argparse.ArgumentParser()
    parser.add_argument("alertness", type=str)
    args = parser.parse_args()

    # check flag validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = args.alertness

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    """""""""""""""""""""""""""""""""""""load x and y"""""""""""""""""""""""""""""""""""""
    print(get_npy_path(alertness,"clause_vector"))
    print(get_npy_path(alertness,"classes"))
    x = np.load(get_npy_path(alertness,"clause_vector"))
    y = np.load(get_npy_path(alertness,"classes"), allow_pickle=True)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""normalise x and encode y"""""""""""""""""""""""""""""
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x, y)
    y_encoded = encode_binary_labels(y) #encode labels
    dump(scaler, get_bin_path(alertness, "scaler"), compress=True)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""smote oversampling to avoid biases"""""""""""""""""""""""""""""""""""""""
    x = list_2d_to_nparray(x_scaled)
    y = np.array(y_encoded)
    x_resampled, y_resampled = SMOTE().fit_resample(x, y)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""do anova feature selection and reshape y to be a 2d array to make it work in keras"""""""""""""
    fvalue_selector = SelectKBest(f_classif, k=12000)
    x_selected = fvalue_selector.fit_transform(x_resampled, y_resampled)
    dump(fvalue_selector, get_bin_path(alertness, "fselector"), compress=True)
    y_2d = list_2d_to_nparray(to_sqaured(y_resampled))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""callbacks and training set test set division"""""""""""""""""""""""""""""""
    X_train, X_test, y_train, y_test = train_test_split(x_selected, y_2d, test_size=0.2, random_state=42)
    sensitive_callback = SensitivitySpecificityCallback((X_test, y_test))
    mcp_save = ModelCheckpoint(get_bow_model_path(alertness), save_best_only=True, monitor='val_loss', mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print(X_test)
    print(y_test)
    # go_baseline_training(X_train, X_test, y_train, y_test, callback=[sensitive_callback, mcp_save, earlyStopping])
    go_training(X_train, X_test, y_train, y_test, callback=[sensitive_callback, mcp_save, earlyStopping])

    # """"""""""""""""""""""" evaluate model """""""""""""""""""""""""""""""""
    # model = load_model('10000_12/best_model.hdf5')
    # in_file = sys.argv[1]
    # with open(in_file) as file:
    #     lines = file.readlines()
    #     for word in lines:
    #         print(word)
    #         break
    # # scores = model.evaluate(x=X_test, y=y_test, verbose=1)
    # # print(round(scores[1]*100,2))
    # # y_pred = np.argmax(model.predict(X_test), axis=1)
    # # print(y_pred)
    # # y_test = np.argmax(y_test, axis=1)
    # # print(classification_report(y_pred,y_test))
    # # print(confusion_matrix(y_test, y_pred))
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""