import numpy as np
import pandas as pd
from create_vocab import list_2d_to_nparray

vocab = np.load('vocab.npy')
clauses = np.load('clauses.npy', allow_pickle=True)
clause_vector = []

clause_counter = 0
for line in clauses:
    print("proceeding...{}/{}".format(clause_counter,len(clauses)))
    temp = [0] * len(vocab)
    for i in range(len(vocab)):
        if vocab[i] in line:
            temp[i] += 1
    clause_vector.append(temp)
    clause_counter += 1

clause_vector = list_2d_to_nparray(clause_vector)
print(clause_vector)
np.save('clause_vector_ex.npy',clause_vector)


