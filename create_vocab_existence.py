import pandas as pd
import numpy as np
import itertools

data = pd.read_csv("data.csv")
clauses = data["clause"]
classes = data["class"]
clauses_by_word = []

def list_2d_to_nparray(list_2d):
    new_array = []
    for line in list_2d:
        new_line = np.asarray(line)
        new_array.append(new_line)
    return np.asarray(new_array)

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
np.save('classes.npy', classes)
np.save('clauses.npy', clauses_by_word)
np.save('vocab.npy', vocabulary)

