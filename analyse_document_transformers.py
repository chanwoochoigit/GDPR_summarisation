import argparse
import math
import sys
from transformers import BertTokenizer
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, add_special_tokens, check_shape_compliance, get_tfm_model_path, take_input
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from transformers import TFBertForSequenceClassification

def formatise_bert_input(clauses):

    max_length = 250
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    token_type_ids = [0] * max_length   #token types are zero because we are doing text classification not QA
    # labels = [3] * max_length

    """""""""instances"""""""""
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    # label_list = []

    for i in range(len(clauses)):
        # print("processing...{}/{}".format(i,len(clauses)))
        # create input ids
        cls_tokenised = tokeniser.tokenize(clauses[i])
        input_ids = tokeniser.convert_tokens_to_ids(cls_tokenised)
        input_ids_padded = input_ids + ([0] * (max_length - len(input_ids)))

        #create attention mask
        # attention_mask = [1] * len(input_ids)   #give more focus on non-padded tokens
        # attention_mask = attention_mask + ([0] * (max_length - len(input_ids)))

        #update each list
        input_ids_list.append(input_ids_padded)
        # token_type_ids_list.append(token_type_ids)
        # attention_mask_list.append(attention_mask)
        # label_list.append([labels[i]])

    print(check_shape_compliance(input_ids_list))
    # print(check_shape_compliance(token_type_ids_list))
    # print(check_shape_compliance(attention_mask_list))

    # return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list)).map(map_to_dict)
    # return tf.data.Dataset.from_tensor_slices((input_ids_list)).map(map_to_dict)
    return tf.convert_to_tensor(input_ids_list)

"""""""""""""""helper function to map bert inputs to dictionary format"""""""""""""""
def map_to_dict(input_ids):
    return {
            "input_ids": input_ids
            # "token_type_ids": token_type_ids,
            # "attention_mask": attention_mask,
            }
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def analyse(json_string):
    # json_object example:
    # {
    #     "alertness": "alice",
    #     "title": "verizon",
    #     "text": "blah blah blah"
    # }
    """""""""""""""""""""""""""""""check and convert json string to json object"""""""""""""""""""""""""""""""
    json_object = take_input(json_string)

    """""""""""""""""""""""""""""""""""take and format"""""""""""""""""""""""""""""""""""
    # check parameter validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = json_object['alertness']

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    policy_text = json_object['text']
    if policy_text == '':
        sys.exit("Text is empty!")

    input_clauses = []
    for line in policy_text.split('\n'):
        if line != '' and line.isspace() is False:
            input_clauses.append(line)

    print(input_clauses[7])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""tokenise and pad clauses"""""""""""""""""""""""""""""""""
    padded_clauses = formatise_bert_input(input_clauses)

    """""""""""""""""""""""""""""""""""load transformers model"""""""""""""""""""""""""""""""""""
    model = load_model(get_tfm_model_path(alertness))
    for line in padded_clauses:
        print(line)
        print(model(line))
        break
    # # predictions = model.predict(x=padded_clauses)
    # print(predictions)
