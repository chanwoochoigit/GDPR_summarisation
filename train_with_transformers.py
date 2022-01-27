import argparse
import math
import sys
import sentencepiece
from numpy import random
from transformers import BertTokenizer
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, add_special_tokens, check_shape_compliance, get_tfm_model_path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from transformers import TFBertForSequenceClassification

def formatise_bert_input(clauses, labels, max_length):

    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    token_type_ids = [0] * max_length   #token types are zero because we are doing text classification not QA

    """""""""instances"""""""""
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for i in range(len(clauses)):
        # print("processing...{}/{}".format(i,len(clauses)))
        # create input ids
        cls_tokenised = tokeniser.tokenize(clauses[i])
        input_ids = tokeniser.convert_tokens_to_ids(cls_tokenised)
        input_ids_padded = input_ids + ([0] * (max_length - len(input_ids)))

        #create attention mask
        attention_mask = [1] * len(input_ids)   #give more focus on non-padded tokens
        attention_mask = attention_mask + ([0] * (max_length - len(input_ids)))

        #update each list
        input_ids_list.append(input_ids_padded)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        label_list.append([labels[i]])

    print(check_shape_compliance(input_ids_list))
    print(check_shape_compliance(token_type_ids_list))
    print(check_shape_compliance(attention_mask_list))

    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_to_dict)

"""""""""""""""helper function to map bert inputs to dictionary format"""""""""""""""
def map_to_dict(input_ids, attention_mask, token_type_ids, label):
    return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            }, label
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


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
    clauses_with_special_tokens = add_special_tokens(clauses)
    y = data["class"]
    y = list(map(lambda x: 1 if x == "worth_reading" else 0, y))
    max_length = 250 # just set max length to be 250
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    X_train, X_test, y_train, y_test = train_test_split(clauses_with_special_tokens, y, test_size=0.2)

    """""""""""""""""""""""""""tokenise clauses with pretrained BERT tokeniser"""""""""""""""""""""""""""
    batch_size=16
    bert_input_train = formatise_bert_input(X_train, y_train, max_length).shuffle(10000).batch(batch_size=batch_size)
    bert_input_test = formatise_bert_input(X_test, y_test, max_length).batch(batch_size=batch_size)

    print(bert_input_train)
    print(bert_input_test)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""set hyper-parameters and compile the model"""""""""""""""""""""""""""""""""
    num_epochs = 1
    learning_rate = 2e-5

    #init model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimiser, loss=loss, metrics=[metric])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # """""""""""""""""""""""""""""""train model using the bert tokenised dataset"""""""""""""""""""""""""""""""
    # bert_history = model.fit(bert_input_train, epochs=num_epochs, validation_data=bert_input_test)
    #
    # """""""""""""""""""""""""""""""""""""""""save model and evaluate"""""""""""""""""""""""""""""""""""""""""
    # # model.save(get_tfm_model_path(alertness), save_format='tf')
    # save_model(model, get_tfm_model_path(alertness), overwrite=True, include_optimizer=True, save_format=None,
    # signatures=None, options=None, save_traces=True)
    # with open(get_tfm_model_path(alertness)+"/max_clause_len.txt", 'w') as text_file:
    #     text_file.write("max_length: "+str(max_length))
    #
    # results = model.evaluate(bert_input_test)
    # print(results)
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
