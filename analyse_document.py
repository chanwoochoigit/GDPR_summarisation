import argparse

from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import load_model
import sys
from joblib import load
import numpy as np
import pandas as pd
from create_vocab import list_2d_to_nparray
from utils import get_bow_model_path, get_npy_path, get_bin_path, store_results, read_predictions

"""helper function to convert clause to BOW vector"""

def vectorise_document(clauses, vocab):
    bow_vectors = []
    clause_counter = 0
    for clause in clauses:
        print("processing...{}/{}".format(clause_counter,len(clauses)))
        clause = clause.replace('.','').replace(',','').replace('eg','').replace('(','').replace(')','').split()
        temp = [0] * len(vocab)
        for i in range(len(vocab)):
            for word in clause:
                if vocab[i] in word:
                    temp[i] += 1
        bow_vectors.append(temp)
        clause_counter += 1
    return bow_vectors

def preprocess_document(document, alertness):
    """""""""""""""""""""standardise vectorised document"""""""""""""""""""""
    scaler = load(get_bin_path(alertness, "scaler"))
    document_scaled = scaler.fit_transform(document)
    # print(document_scaled)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""do feature selection on the vectorised test document"""""""""""
    fselector = load(get_bin_path(alertness, "fselector"))
    document_scaled_selected = fselector.transform(document_scaled)
    print(document_scaled_selected.shape)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    return document_scaled_selected


if __name__ == '__main__':

    # take flags
    parser = argparse.ArgumentParser()
    parser.add_argument("alertness", type=str)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    # check flag validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = args.alertness

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    filename = args.filename
    print(filename)

    input_clauses = []

    with open(filename, 'r') as file:
        for line in file:
            if line != '' and line.isspace() is False:
                input_clauses.append(line)

    """""""""""""""""""""""""""""""""""""load vocab, model and clause vector"""""""""""""""""""""""""""""""""""""
    vocab = np.load(get_npy_path(alertness, "vocab"))
    model = load_model(get_bow_model_path(alertness))
    clauses_vector = list_2d_to_nparray(vectorise_document(clauses=input_clauses, vocab=vocab))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""load and preprocess vectorised document text file"""""""""""""""""""""""
    # verizon = np.load('verizon.npy')
    document_processed = preprocess_document(document=clauses_vector, alertness=alertness)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    predictions = model.predict(document_processed)
    results = read_predictions(predictions, "bow")
    print(results.shape)
    store_results(input_clauses, results, filename[:-4], "bow")
