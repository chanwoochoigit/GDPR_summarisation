import glob
from pathlib import Path
import pickle

dir = 'ppdocs'
paths = Path(dir).rglob('*.txt')
num_paths = len(glob.glob('google_places_api_data/*'))
all_clauses = []
for path in paths:
    with open(path, 'r') as f:
        text = f.read().replace('-', ' ').replace('!', '').replace('[', ' ').replace(']', ' ').replace(':', ' ')\
                .replace(';', ' ').replace('(', ' ').replace(')', ' ').replace('_', ' ').replace('   ', ' ')\
                .replace('  ', ' ').replace('    ', ' ').replace('. ', '.').split('\n')
        text.remove('')
        for t in text:
            if t == '':
                continue
            if len(t) <= 3:
                continue
            all_clauses.append(t)

with open('clauses_v2.pkl', 'wb') as f:
    pickle.dump(all_clauses, f)