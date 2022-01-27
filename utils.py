import json

import numpy as np
from sys import exit
import pandas as pd

def list_2d_to_nparray(list_2d):
    new_array = []
    for line in list_2d:
        new_line = np.asarray(line)
        new_array.append(new_line)
    return np.asarray(new_array)

def encode_binary_labels(y):
    encoded_labels = []
    labels = []
    [labels.append(x) for x in y if x not in labels]
    print(labels)
    for label in y:
        if label == labels[0]:
            encoded_labels.append(0)
        elif label == labels[1]:
            encoded_labels.append(1)
        else:
            print(label)
            exit("wrong label input!")
    return encoded_labels

def get_npy_path(alertness, filename):
    return "training_data/"+alertness+"/"+filename+"_"+alertness+".npy"

def get_bin_path(alertness, name):
    return "training_data/"+alertness+"/"+name+"_"+alertness+".bin"

def get_bow_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bow"

def get_bert_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bert"

def get_tfm_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_tfm"

def get_max_length_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bert/max_clause_len.txt"

def get_max_length(x_data):
    max_length = 0
    for x in x_data:
        if len(x) > max_length:
            max_length = len(x)
    return max_length

def max_length_padding(tokenised_clauses, max_length=0):
    """""""""""""""""""""""""""""""""""pad clauses with 0 to make them equal in length"""""""""""""""""""""""""""""""""""
    if max_length == 0:
        max_length = get_max_length(tokenised_clauses)
    else:
        max_length = max_length
    padded_clauses = []
    for clause in tokenised_clauses:
        padded_clause = clause
        while len(padded_clause) < max_length:
            padded_clause.append(0)
        padded_clauses.append(padded_clause)
    padded_clauses = list_2d_to_nparray(padded_clauses)
    print(padded_clauses.shape)

    return padded_clauses

def tokenise_clauses(clause, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(clause))

def add_special_tokens(x_data):
    new_x = []
    for x in x_data:
        new_x.append('[CLS] ' + x + ' [SEP]')

    return new_x

def read_predictions(predictions, mode):
    results = []
    standard_counter = 0
    worth_counter = 0
    if mode == "bow":
        for pred in predictions:
            if pred[0] > pred[1]:
                results.append("standard_trivial")
                standard_counter += 1
            elif pred[0] < pred[1]:
                results.append("worth_reading")
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_important = round(worth_counter / (standard_counter + worth_counter),4) * 100
        print("Ratio of worth reading clauses: "+str(how_important)+"%")
    elif mode == "bert":
        for pred in predictions:
            if pred[0] < 0.5:
                results.append("standard_trivial")
                standard_counter += 1
            elif pred[0] > 0.5:
                results.append("worth_reading")
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_important = round(worth_counter / (standard_counter + worth_counter), 4) * 100
        print("Ratio of worth reading clauses: " + str(how_important) + "%")
    else:
        exit("Wrong mode selection!")
    return np.array(results)

def get_standard_ratio(predictions, mode):
    standard_counter = 0
    worth_counter = 0
    how_standard = 0
    if mode == "bow":
        for pred in predictions:
            if pred[0] > pred[1]:
                standard_counter += 1
            elif pred[0] < pred[1]:
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_important = round(worth_counter / (standard_counter + worth_counter),4) * 100
        print("Ratio of worth reading clauses: "+str(how_important)+"%")
    elif mode == "bert":
        for pred in predictions:
            if pred[0] < 0.5:
                standard_counter += 1
            elif pred[0] > 0.5:
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_standard = round(standard_counter / (standard_counter + worth_counter), 4) * 100
        print("Ratio of standard clauses: " + str(how_standard) + "%")
    else:
        exit("Wrong mode selection!")
    return how_standard

def store_results(document, results, filename, mode):
    classified_df = pd.DataFrame()
    classified_df['clause'] = document
    classified_df['class'] = results
    classified_df.to_csv("results/"+filename+'_classified_result_'+mode+'.csv')

def check_shape_compliance(data):
    len_list = []
    for line in data:
        len_list.append(len(line))
    # print(len_list)
    return all(x == len_list[0] for x in len_list)

def take_input(potential_json):
    try:
        json_object = json.loads(potential_json)
        return json_object
    except ValueError:
        raise ValueError("The entered string is not a valid json string!")
