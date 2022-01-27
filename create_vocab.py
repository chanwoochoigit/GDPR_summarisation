import pandas as pd
import numpy as np
import itertools
import argparse
import sys
from utils import get_npy_path, list_2d_to_nparray

if __name__ == '__main__':
    #take flags
    parser = argparse.ArgumentParser()
    parser.add_argument("alertness",type=str)
    args = parser.parse_args()

    #check flag validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = args.alertness

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    data_path = "training_data/"+alertness+"/data_"+alertness+".csv"

    data = pd.read_csv(data_path)
    clauses = data["clause"]
    classes = data["class"]
    clauses_by_word = []
    print(classes)
    clauses_counter = 0
    for line in clauses:
        print("proceeding...{}/{}".format(clauses_counter, len(clauses)))
        new_line = line.replace('.','').replace(',','').replace('eg','').replace('(','').replace(')','').split()
        clauses_by_word.append(new_line)
        clauses_counter += 1

    vocabulary = []
    cbw_counter = 0
    for line in clauses_by_word:
        print("proceeding...{}/{}".format(cbw_counter, len(clauses_by_word)))
        temp = []
        [temp.append(x) for x in line if x not in temp]
        vocabulary.append(np.array(temp))
        cbw_counter += 1


    vocabulary = np.array(list(itertools.chain.from_iterable(vocabulary)))
    print(vocabulary.shape)

    clauses_by_word = list_2d_to_nparray(clauses_by_word)
    print(clauses_by_word.shape)
    np.save(get_npy_path(alertness, "classes"), classes)
    np.save(get_npy_path(alertness, "clauses"), clauses_by_word)
    np.save(get_npy_path(alertness, "vocab"), vocabulary)

