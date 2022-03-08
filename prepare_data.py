import glob
from pathlib import Path
import pickle
import itertools

dir = 'ppdocs'
paths = Path(dir).rglob('*.txt')
num_paths = len(glob.glob('google_places_api_data/*'))
raw_text = []
for path in paths:
    with open(path, 'r') as f:
        raw_text.append(f.read())

def split_sentences(sentences):
    splitted = []
    for i, s in enumerate(sentences):
        #remove unsupported punctuations
        splitted.append(s.replace('-',' ').replace('!','').replace('[',' ').replace(']',' ').replace(':',' ').replace(';',' or ')\
                            .replace('(a)',' ').replace('(b)','').replace('(c)','').replace('(d)','').replace('(e)','') \
                            .replace('(','').replace(')','').replace('\n','.')\
                            .replace('\u2019',' ').replace('\u2013',' ').replace('\u2014',' ').replace('\u201d',' ')
                            .replace('\u201c',' ').replace('\u2018', ' ').replace('\u202f', ' ').replace('\u00e0', ' ')
                            .replace('\u00e9',' ').replace('\u00a0', ' ').replace('(', ' ').replace(')',' ').replace('_',' ')
                            .replace('   ',' ').replace('  ',' ').replace('    ',' ').replace('U.S.','USA').replace('E.U.','eu').replace('e.g.','for example,')\
                            .replace('(Japan)','japan')
                            .replace('. ','.').replace('.','. ')
                            .split('. '))

    splitted = list(itertools.chain.from_iterable(splitted))

    formatted = []
    for sentence in splitted:
        if sentence == '': continue
        else:
            if '.' not in sentence:
                formatted.append(str(sentence)+'.')
            else:
                formatted.append(str(sentence))

    return formatted

all_clauses = split_sentences(raw_text)

with open('clauses_v2.pkl', 'wb') as f:
    pickle.dump(all_clauses, f)