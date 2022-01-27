import argparse
import sys
import numpy as np
import pandas as pd
from utils import list_2d_to_nparray, get_npy_path


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

    vocab = np.load(get_npy_path(alertness, "vocab"))
    clauses = np.load(get_npy_path(alertness, "clauses"), allow_pickle=True)
    clause_vector = []

    clause_counter = 0
    for line in clauses:
        print("proceeding...{}/{}".format(clause_counter,len(clauses)))
        temp = [0] * len(vocab)
        for i in range(len(vocab)):
            for word in line:
                if vocab[i] in word:
                    temp[i] += 1
        clause_vector.append(temp)
        clause_counter += 1

    clause_vector = list_2d_to_nparray(clause_vector)
    print(clause_vector)
    np.save(get_npy_path(alertness, "clause_vector"),clause_vector)


