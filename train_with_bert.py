import argparse
import math
import sys

import bert
import sentencepiece
from numpy import random
from tensorflow.keras import layers
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, get_max_length, max_length_padding, tokenise_clauses
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


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

    """""""""""""""""""""""""""""""""""load data and assign variables"""""""""""""""""""""""""""""""""""
    data = pd.read_csv("training_data/"+alertness+"/data_"+alertness+".csv")
    clauses = data["clause"]
    y = data["class"]
    y = np.array(list(map(lambda x: 1 if x == "worth_reading" else 0, y)))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""download and load bert tokeniser + vocabulary"""""""""""""""""""""""""""
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    """""""""""""""""""""""""""""""""""""""tokenise clauses with BERT tokeniser"""""""""""""""""""""""""""""""""""""""
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    tokenised_clauses = [tokenise_clauses(clause, tokenizer) for clause in clauses]
    max_length = get_max_length(tokenised_clauses)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    """""""""""""""""""""""""""""""""""""""""""smote oversampling to avoid biases"""""""""""""""""""""""""""""""""""""""
    padded_clauses = max_length_padding(tokenised_clauses)
    x_resampled, y_resampled = SMOTE().fit_resample(padded_clauses, y)
    print(x_resampled.shape)
    print(y_resampled.shape)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""hyper-parameters"""""""""""""
    BATCH_SIZE = 16

    VOCAB_LENGTH = len(tokenizer.vocab)
    EMB_DIM = 200
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 2

    DROPOUT_RATE = 0.3

    NUM_EPOCHS = 30
    """"""""""""""""""""""""""""""""""""""""""

    classifier = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                            embedding_dimensions=EMB_DIM,
                            cnn_filters=CNN_FILTERS,
                            dnn_units=DNN_UNITS,
                            model_output_classes=OUTPUT_CLASSES,
                            dropout_rate=DROPOUT_RATE)

    """"""""""""""""""""""""""""compile classes; here we only use binary as there are 2 classes"""""""""""""""""""""""""""""
    if OUTPUT_CLASSES == 2:
        classifier.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])
    else:
        classifier.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=["sparse_categorical_accuracy"])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""divide training & test set and train"""""""""""""""""""""""""""""""""""""""""
    X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    classifier.fit(x=X_train, y=y_train, epochs=NUM_EPOCHS)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""save model and max length"""""""""""""""""""""""""""""""""""""""""""""
    classifier.save(get_bert_model_path(alertness), save_format='tf')
    with open(get_bert_model_path(alertness)+"/max_clause_len.txt", "w") as text_file:
        text_file.write("max_length: "+str(max_length))

    results = classifier.evaluate(x=X_test, y=y_test)
    print(results)
