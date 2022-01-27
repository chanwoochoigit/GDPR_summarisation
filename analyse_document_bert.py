import argparse
import json
import sys
import pandas as pd
import bert
# import sentencepiece
# from numpy import random
# from tensorflow.keras import layers
import tensorflow_hub as hub
# import pandas as pd
# import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, tokenise_clauses, get_max_length_path, max_length_padding, read_predictions, \
    store_results, take_input, get_standard_ratio


def clauses_to_json_string(clauses, worth_indices):
    input_clauses = clauses
    for i in range(len(input_clauses)):
        if i in worth_indices:
            input_clauses[i] = "***"+input_clauses[i]

    result_df = pd.DataFrame(input_clauses, columns=['clause'])
    result_json = result_df.to_json()
    try:
        print("Successfully save final result to json.")
        print(result_json)
        return result_json
    except:
        print("I/O error: Failed to save final_result to json! Get this shit sorted.")

def analyse(json_string):

    # json_object example:
    # {
    #     "alertness": "alice",
    #     "title": "verizon",
    #     "text": "blah blah blah"
    # }
    """""""""""""""""""""""""""""""check and convert json string to json object"""""""""""""""""""""""""""""""
    json_object = take_input(json_string)

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

    """""""""""""""""""""""""""""""get maximum clause length for test data"""""""""""""""""""""""""""""""
    with open(get_max_length_path(alertness), "r") as max_len_file:
        max_len = int(max_len_file.read()[-3:])
    print(max_len)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""download and load bert tokeniser + vocabulary"""""""""""""""""""""""""""
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""tokenise clauses with BERT tokeniser"""""""""""""""""""""""""""""""""""""""
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    tokenised_clauses = [tokenise_clauses(clause, tokenizer) for clause in input_clauses]
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #padd data
    padded_clauses = max_length_padding(tokenised_clauses, max_length=max_len)

    model = load_model(get_bert_model_path(alertness))
    predictions = model.predict(x=padded_clauses)
    print(predictions)
    results = read_predictions(predictions, "bert")
    print(results)
    store_results(input_clauses, results, json_object['title'].replace(' ',''), "bert")

    standard_ratio = get_standard_ratio(predictions, "bert")

    """""""""""""""get indices for worth reading clauses"""""""""""""""
    worth_indices = []
    for i in range(len(results)):
        if results[i] == "worth_reading":
            worth_indices.append(i)

    clauses_json = json.loads(clauses_to_json_string(input_clauses, worth_indices))
    clauses_json['standard_ratio'] = standard_ratio
    print(clauses_json)
    return clauses_json


