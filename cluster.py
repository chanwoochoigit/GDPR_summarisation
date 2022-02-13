import itertools
import pickle
import json
import time
from collections import defaultdict
from math import log2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import nltk

class Clustering():

    def list_to_str(self, tokens_list):
        sentences = []
        for line in tokens_list:
            str = ''
            for token in line:
                str += token+' '
            sentences.append(str)

        return sentences

    def do_sentenceBERT(self, sentences):
        model = SentenceTransformer('stsb-mpnet-base-v2')
        sentence_embeddings = model.encode(sentences)
        # print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
        # print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])
        sentence_dict = {}
        sentence_dict_reverse = {}
        for raw, emb in zip(sentences, sentence_embeddings):
            sentence_dict[raw] = emb.tolist()
            sentence_dict_reverse[str(emb.tolist())] = raw

        with open('stnce_embedding.json', 'w') as f:
            json.dump(sentence_dict, f)

        with open('stnce_embedding_reverse.json', 'w') as f:
            json.dump(sentence_dict_reverse, f)

    def cluster_decode(self, X, pred, num_clusters):
        sent_cluster_dict = {}
        for x, p in zip(X, pred):
            sent_cluster_dict[str(sent_dict_reverse[str(x.tolist())])] = str(p)
            # print(str(sent_dict_reverse[str(x.tolist())])+': '+str(p))

        #decode encoded sentences and save to a dict -> dict[cluster_num] = decoded_vector
        cluster_dict = {}
        for i in range(num_clusters):
            print('processing...{}/{}'.format(i, num_clusters))
            temp = []
            for k, v in sent_cluster_dict.items():
                if str(v) == str(i):
                    temp.append(k)
            cluster_dict[i] = temp

        return sent_cluster_dict

    def do_kmeans(self, sentences, num_clusters):
        X = np.asarray(sentences)

        #normalise vector to make cosine similarity effect
        normed_X = normalize(X, axis=1, norm='l1')
        # K-Means Parameters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(normed_X)

        # collect centers and find the closest vector to get approximate center vector
        # this process is required because a center vector is a mean of other vectors and not be able to be decoded
        centers = kmeans.cluster_centers_
        centers_decodable = {}
        start = time.time()
        for i, center in enumerate(centers):
            print('finding appx. center for cluster {}/{} ... running time: {} seconds'.format(i, num_clusters, round(time.time()-start,4)))
            distance = 1e+10    #base distance: to be updated so a random large number is given
            for sent in sentences:
                temp_dist = euclidean(center, sent)
                if temp_dist < distance:
                    distance = temp_dist
                    centers_decodable[i] = sent

        pred = kmeans.predict(normed_X)

        #decode clustered sentences to plain text
        cluster_dict = self.cluster_decode(X, pred, num_clusters)

        with open('sent_cluster.json', 'w') as f:
            json.dump(cluster_dict, f)


    # perform elbow method to find optimal k: returns WSS score for k values from 1 to kmax
    def do_elbow(self, sentences, kmax):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l1')

        sse = []
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(normed_X)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(normed_X)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(normed_X)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (normed_X[i, 0] - curr_center[0]) ** 2 + (normed_X[i, 1] - curr_center[1]) ** 2

            sse.append(curr_sse)

        y = sse
        x = range(len(y))
        plt.plot(x, y)
        plt.savefig('elbow.png')
        plt.show()


    def do_silhoulette(self, sentences, kmax):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l1')

        sil = []
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(normed_X)
            labels = kmeans.labels_
            sil.append(silhouette_score(normed_X, labels, metric='euclidean'))

        y = sil
        x = range(len(y))
        plt.plot(x, y)
        plt.savefig('silhouette.png')
        plt.show()

class EvalCluster():

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    def create_corpus(self):
        with open('sent_cluster.json', 'r') as f:
            clusters = json.load(f)
        docs_processed = self.init_nd_dict()
        for c in clusters.keys():
            for i, doc in enumerate(clusters[c]):
                # processed = self.stem_data(self.remove_stopwords(doc.replace('.','').replace(',','')))
                processed = self.remove_stopwords(doc.replace('.', '').replace(',', ''))
                docs_processed[c][i] = processed
        with open('corpus.json', 'w') as f:
            json.dump(docs_processed, f)

    def load_corpus(self):
        with open('corpus.json', 'r') as f:
            corpus = json.load(f)
        return corpus

    def str_to_words(self, string):
        words = string.split(' ')
        word_list = []
        punctuations = [',','.', '?', ';',':','-','_',')','(',']','[','`','~','"','<','>']
        for word in words:
            if word not in punctuations:    # exclude punctuations and meaningless marks when appending
                word_list.append(word)

        return word_list

    def remove_stopwords(self, words):
        stopwords = []
        with open('stopwords.txt', 'r') as f:
            temp = f.readlines()
        for word in temp:
            if word != ' ' or word != '\n':
                stopwords.append(word.replace('\n',''))
        words_nostop = []
        word_list = self.str_to_words(words)
        [words_nostop.append(x) for x in word_list if x not in stopwords and x != '']
        return words_nostop

    def stem_data(self, words_preprocessed):
        nltk.download('punkt')

        ps = nltk.PorterStemmer()
        words_stemmed = []
        for word in words_preprocessed:
            words_stemmed.append(ps.stem(word))
        return words_stemmed

    # calculate mutual information given all 4 counts
    def calc_mi(self, term, cls):
        N11, N10, N01, N00 = self.get_Ns(term, cls)

        N = N11 + N10 + N01 + N00

        try:
            aa = (N11 / N) * log2((N * N11) / ((N11 + N10) * (N01 + N11)))
        except:
            aa = 0
        try:
            bb = (N01 / N) * log2((N * N01) / ((N01 + N00) * (N01 + N11)))
        except:
            bb = 0
        try:
            cc = (N10 / N) * log2((N * N10) / ((N10 + N11) * (N10 + N00)))
        except:
            cc = 0
        try:
            dd = (N00 / N) * log2((N * N00) / ((N00 + N01) * (N10 + N00)))
        except:
            dd = 0

        return aa + bb + cc + dd

    # get counts to calculate mutual information and Chi-squared
    def get_Ns(self, term, cls):
        corpus = self.load_corpus()
        classes = corpus.keys()

        # find "non-current" class
        c0 = []  # len(c0) is always 2
        for c in classes:
            if c != cls:
                c0.append(c)

        N11, N10, N01, N00 = 0, 0, 0, 0

        # investigate document in the given class
        for doc in corpus[cls].keys():
            curr_doc = corpus[cls][doc]
            if term in curr_doc:
                N11 += 1
            elif term not in curr_doc:
                N01 += 1

        # investigate documents in other classes
        for c in c0:
            for doc in corpus[c].keys():
                curr_doc = corpus[c][doc]
                if term in curr_doc:
                    N10 += 1
                elif term not in curr_doc:
                    N00 += 1

        return N11, N10, N01, N00

    def run_calculation(self, num_classes):
        corpus = self.load_corpus()
        result = self.init_nd_dict()

        for i, cls in enumerate(corpus.keys()):
            for doc in corpus[cls]:
                print('class: {}/{}---------------------------------------------------'.format(i+1, num_classes))
                print('calculating mutual information...{}/{}'.format(doc, len(corpus[cls].keys())))
                for word in corpus[cls][doc]:
                    score = self.calc_mi(word, cls)
                    result[word][cls] = score

        with open('{}.json'.format('mi_scores'), 'w') as f:
            json.dump(result, f)

        return result

    def format_ranked_result(self, result_dict):
        result = self.init_nd_dict()
        for i, item in enumerate(result_dict.items()):
            term = item[0]
            score = item[1]
            result[term] = round(score,4)
            if i > 30 :
                break
        return result

    def sort_dict_by_value(self, dict_to_sort):
        return dict(sorted(dict_to_sort.items(), key=lambda item: item[1], reverse=True))

    def sort_result(self, num_classes):
        with open('mi_scores.json', 'r') as f:
            to_display = json.load(f)
        to_sort = self.init_nd_dict()
        for word in to_display.keys():
            for corpus in to_display[word]:
                score = to_display[word][corpus]
                to_sort[corpus][word] = score

        result = self.init_nd_dict()

        for i in range(num_classes):
            result[i] = self.format_ranked_result(self.sort_dict_by_value(to_sort[str(i)]))

        with open('ranked_result_cluster.json', 'w') as f:
            json.dump(result, f)

    def validate_result(self, num_keywords, max_sentences):
        superwords = self.init_nd_dict()

        with open('ranked_result_cluster.json', 'r') as f:
            results = json.load(f)

        # choose 5 "strongest" words and store them in a dict
        for i in list(results.keys()):
            for j in range(num_keywords):
                superwords[i][j] = list(results[str(i)].keys())[j]
        # print(superwords)

        with open('sent_cluster.json', 'r') as f:
            clusters = json.load(f)

        super_sentences = self.init_nd_dict()

        for cluster in clusters.keys():             #for each cluster
            sentences = clusters[cluster]
            temps = [[], [], [], [], []]

            for s in sentences:
                for i in range(num_keywords):
                    if superwords[cluster][i] in s and s not in temps[i]:   # if each superword is in any of the sentences in the same cluster
                        temps[i].append(s)                                  # where the superword belongs, append it to a list

            for i in range(num_keywords):
                super_sentences[cluster][superwords[cluster][i]] = temps[i]  # save the list of sentences in a dict

        self.format_supersentences(superwords, super_sentences, max_sentences)  #format the "supersentences" to be human-readable to a file

    def format_supersentences(self, superwords, super_sentences, max_sentences):
        with open('supersentences_per_cluster.txt', 'a') as f:
            for cluster in super_sentences.keys():
                f.write('cluster ' + str(cluster) + ':\n')

                for i in range(max_sentences):
                    f.write('\tword {} "{}":'.format(i+1, superwords[cluster][i]) + '\n')
                    target_word = superwords[cluster][i]
                    sentences = super_sentences[cluster][target_word]
                    for j, s in enumerate(sentences):
                        if j == max_sentences:
                            break
                        f.write('\t\t{}: "{}"\n'.format(j+1, s))

                f.write('\n')

class UtilityFunct():

    def split_sentences(self, sentences):
        splitted = []
        for s in sentences:
            #remove unsupported punctuations
            splitted.append(s.replace('-',' ').replace('!','').replace('[',' ').replace(']',' ').replace(':',' ').replace(';',' ')\
                            .replace('(a)',' ').replace('(b)','').replace('(c)','').replace('(d)','').replace('(e)','') \
                            .replace('\u2019',' ').replace('\u2013',' ').replace('\u2014',' ').replace('\u201d',' ')
                            .replace('\u201c',' ').replace('\u2018', ' ').replace('\u202f', ' ').replace('\u00e0', ' ')
                            .replace('\u00e9',' ').replace('\u00a0', ' ').replace('(', ' ').replace(')',' ').replace('_',' ')
                            .replace('   ',' ').replace('  ',' ').replace('    ',' ').replace('U.S.','USA').replace('E.U.','eu').replace('e.g.','for example,')
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


"""""""""""""use alice clauses for experiment"""""""""""""
knn = Clustering()
# alice_csv = pd.read_csv('training_data/alice/data_alice.csv')
# alice_clause = alice_csv['clause'].to_list()
#
# uf = UtilityFunct()
#
# sentences = uf.split_sentences(alice_clause)
#
# knn.do_sentenceBERT(sentences)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""load embedded sentences to be processed"""""""""
with open('stnce_embedding.json', 'r') as f:
    sent_dict = json.load(f)
with open('stnce_embedding_reverse.json', 'r') as f:
    sent_dict_reverse = json.load(f)

embedded_sentences = list(sent_dict.values())
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""perform k-means clustering"""""""""""""
knn.do_kmeans(embedded_sentences, 20)
# knn.do_elbow(embedded_sentences,150)
# knn.do_silhoulette(embedded_sentences,100)
""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""evaluate k-means"""""""""""""""""
eval = EvalCluster()
# eval.create_corpus()
# eval.run_calculation(20)
# eval.sort_result(num_classes=20)
# eval.validate_result(num_keywords=5, max_sentences=5)
""""""""""""""""""""""""""""""""""""""""""""""""""