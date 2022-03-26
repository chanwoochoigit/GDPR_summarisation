import itertools
import logging
import math
import pickle
import json
import random
import re
import time
from collections import defaultdict
from math import log2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rouge
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch, \
                            AffinityPropagation, MeanShift, OPTICS, \
                            AgglomerativeClustering
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import hdbscan
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from logging.handlers import RotatingFileHandler

# set logging
log_name = 'cluster.py.log'
logging.basicConfig(filename=log_name, format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger()
handler = RotatingFileHandler(log_name, maxBytes=1048576)
log.addHandler(handler)
log.propagate=False

class Clustering():

    def list_to_str(self, tokens_list):
        sentences = []
        for line in tokens_list:
            str = ''
            for token in line:
                str += token+' '
            sentences.append(str)

        return sentences

    # process chunks of sentences, remove unsupported punctuation and split them
    def do_sentenceBERT(self, sentences, labels):
        model = SentenceTransformer('snt_tsfm_model/')

        worth_sentences = []   #only get important sentences
        for i, s in enumerate(sentences):
            if labels[i] == 'worth_reading':
                worth_sentences.append(s)

        sentence_embeddings = model.encode(worth_sentences)
        # log.info('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
        # log.info('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

        sentence_dict = {}
        sentence_dict_reverse = {}
        for raw, emb in zip(worth_sentences, sentence_embeddings):
            sentence_dict[raw] = emb.tolist()
            sentence_dict_reverse[str(emb.tolist())] = raw

        #create sentence embedding dict[plain_text] = sentence_vector
        with open('stnce_embedding.json', 'w') as f:
            json.dump(sentence_dict, f)

        #create reversed sentence embedding dict[sentence_vector] = plain_text
        with open('stnce_embedding_reverse.json', 'w') as f:
            json.dump(sentence_dict_reverse, f)

    #convert sentence vectors to plain text and save it to dict[cluster] = sentence
    def cluster_decode(self, X, pred, num_clusters):
        sent_cluster_dict = {}
        with open('stnce_embedding_reverse.json', 'r') as f:
            sent_dict_reverse = json.load(f)

        for x, p in zip(X, pred):   # convert sentence vector to plain text, and put it in a dictionary
                                    # dict[plain_text] = cluster
            try:
                sent_cluster_dict[str(sent_dict_reverse[str(x.tolist())])] = str(p)
                # log.info(str(sent_dict_reverse[str(x.tolist())])+': '+str(p))
            except AttributeError:  # decoding list of centroids
                sent_cluster_dict[str(sent_dict_reverse[str(x)])] = str(p)

        #decode encoded sentences and save to a dict -> dict[cluster_num] = decoded_vector
        cluster_dict = {}
        for i in range(num_clusters):
            log.info('saving decoded sentence vectors ... {}/{}'.format(i, num_clusters))
            temp = []
            for k, v in sent_cluster_dict.items():
                if str(v) == str(i):
                    temp.append(k)
            cluster_dict[i] = temp

        return cluster_dict

    # generate key for pca-ed sentence vectors to decode afterwards
    def generate_pca_key(self, principle_comp):
        # create pseudo-unique keys to create kv table
        return str(principle_comp[0]/principle_comp[1]*principle_comp[2])[3:]

    # return the nearest point in a dataset from a given point
    def find_nearest_point(self, data, point):
                                #calculate distance from nearest to furthest
        d = euclidean(data[0], point)
        p = None    #init data point outside of for loop

        for i in range(len(data)):
            temp_dist = euclidean(data[i],point)
            if temp_dist < d:
                log.info('updating d1 from {} to {}'.format(d, temp_dist))
                d = temp_dist
                p = data[i]

        return p

    # plot clustering result based on labels(clusters) generated
    def plot_labels(self, df, pred, algorithm):
        u_labels = np.unique(pred)
        for i in u_labels:
            plt.scatter(df[pred == i, 0], df[pred == i, 1], label=i)
        # plt.legend(bbox_to_anchor=(1.14,1.05), loc='upper right')
        plt.savefig('clustering_results/{}.png'.format(algorithm))

    # perform unsupervised clustering and compare results for different clustering algorithms
    def perform_clustering(self, sentences, clusterer, metric, num_clusters=None, pca2vector=None):
        # prepare data
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2') # normalise vector to make cosine similarity effect
        sil = 999

        if clusterer == 'kmeans':
            if num_clusters is not None:
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(normed_X)
                labels = kmeans.predict(normed_X)

                if pca2vector is not None:

                    # get centroids and nearest points for further analysis
                    centroids = kmeans.cluster_centers_
                    pseudo_centers = []
                    for c in centroids:  # get nearest point from each centroid because centroid is a synthesised point
                        pc = self.find_nearest_point(sentences, c)  #pc is short for pseudo-centre
                        pseudo_centers.append(pca2vector[self.generate_pca_key(pc)])

                    # decode cluster centres and write out to a json file
                    cluster_labels = list(range(num_clusters))
                    log.info('pseudo centres length: {}'.format(len(pseudo_centers)))
                    log.info('cluster labels length: {}'.format(len(cluster_labels)))
                    centers_decoded = self.cluster_decode(pseudo_centers, cluster_labels, num_clusters)
                    with open('cluster_centers.json', 'w') as f:
                        json.dump(centers_decoded, f)

                    ## get all sentences clustered to examine if clustering is working fine (by human eyes)
                    # pca_decoded = []
                    # for pca in X:
                    #     pca_decoded.append(pca2vector[self.generate_pca_key(pca)])
                    # # decode clustered sentences to plain text
                    # cluster_dict = self.cluster_decode(pca_decoded, labels, num_clusters)
                    # with open('sentences_clustered.json', 'w') as f:
                    #     json.dump(cluster_dict, f)

                else:
                    'pca2vector dictionary not given! skipping decoding process ...'

            else:
                raise SyntaxError('num_cluster not entered for Kmeans!')

        elif clusterer == 'kmeans_mini':
            if num_clusters is not None:
                minik = MiniBatchKMeans(n_clusters=num_clusters).fit(normed_X)
                labels = minik.predict(normed_X)
            else:
                raise SyntaxError('num_cluster not entered for Kmeans!')

        elif clusterer == 'dbscan':
            dbs = DBSCAN(eps=0.3,min_samples=9,metric=metric)
            labels = dbs.fit_predict(normed_X)

        elif clusterer == 'hdbscan':
            hdb = hdbscan.HDBSCAN(metric=metric,min_samples=5, min_cluster_size=num_clusters)
            labels = hdb.fit_predict(normed_X)

        elif clusterer == 'gaussian':
            gm = GaussianMixture(n_components=num_clusters)
            gm.fit(normed_X)
            labels = gm.predict(normed_X)

        elif clusterer == 'birch':
            if num_clusters is not None:
                birch = Birch(threshold=0.03, n_clusters=num_clusters)
                birch.fit(normed_X)
                labels = birch.predict(normed_X)
            else:
                raise SyntaxError('num_cluster not entered for Birch!')

        elif clusterer == 'affinity':
            aff = AffinityPropagation(damping=0.9,random_state=42)
            aff.fit(normed_X)
            labels = aff.predict(normed_X)

        elif clusterer == 'meanshift':
            ms = MeanShift()
            ms.fit(normed_X)
            labels = ms.predict(normed_X)

        elif clusterer == 'optics':
            optics = OPTICS(eps=0.8, min_samples=10)
            labels = optics.fit_predict(normed_X)

        elif clusterer == 'agglomerative':
            agg = AgglomerativeClustering(n_clusters=num_clusters)
            labels = agg.fit_predict(normed_X)

        elif clusterer is None:
            raise ValueError('please enter a clustering algorithm!')

        else:
            raise ValueError('chosen clusterer not supported on current version!')

        log.info(len(set(labels))) # get number of clusters for algorithms not requiring num_cluster param
        try:
            sil = silhouette_score(normed_X, labels)
            log.info(sil)
        except ValueError:
            log.info('clustering failed miserably! There is only a single cluster.')

        # plot kmeans for further checkup
        self.plot_labels(normed_X, labels, clusterer)


        return sil


    # perform elbow method to find optimal k: returns WSS score for k values from 1 to kmax
    def do_elbow(self, sentences, kmax):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2')

        sse = []
        start = time.time()
        for k in range(1, kmax + 1):
            log.info('finding the optimum cluster number ... {}/{} | running time: {} seconds'.format(k,kmax,round(time.time()-start,4)))
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

    # get cumulative sum to find num of principal components and corresponding variance
    def find_best_ncomp_for_pca(self, sentences):
        pca = PCA()
        pca.fit(sentences)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
        plt.xlabel('Components')
        plt.ylabel('Explained Variances')
        plt.show()

    # perform pca on sentence vectors
    # returns: pca-ed sentence vectors and a dict of K(key generated from pca):V(sentence vector)
    def do_pca(self, n_components, sentences):

        pca = PCA(n_components=n_components, random_state=42)
        pca_sentences = pca.fit_transform(sentences)

        #create a dictionary that converts principle components to sentence vectors for later
        pca2vector = {}
        for pca, vector in zip(pca_sentences, sentences):
            pca2vector[self.generate_pca_key(pca)] = vector

        return pca_sentences, pca2vector

    # perform silhouette analysis to find optimal value of k
    def do_silhouette(self, sentences, kmax):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2')

        sil = []
        start = time.time()
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            log.info('finding the optimum cluster number ... {}/{} | running time: {} seconds'.format(k,kmax,round(time.time()-start,4)))
            kmeans = KMeans(n_clusters=k).fit(normed_X)
            labels = kmeans.labels_
            sil.append(silhouette_score(normed_X, labels, metric='euclidean'))

        y = sil
        x = range(len(y))
        plt.plot(x, y)
        plt.savefig('silhouette.png')
        plt.show()

    # perform silhouette analysis for defined single value k
    def do_silhouette_single(self, sentences, k):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2')

        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        kmeans = KMeans(n_clusters=k).fit(normed_X)
        labels = kmeans.labels_
        return silhouette_score(normed_X, labels, metric='euclidean')

    def format_cluster_centres(self):
        with open('cluster_centers.json', 'r') as f:
            centres = json.load(f)
        with open('supersentences_from_cluster_centres.txt', 'w') as f:
            for key in centres.keys():
                f.write('{}:\n'.format(key))
                for sentence in centres[key]:
                    f.write('{}\n'.format(sentence))

class EvalCluster():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    def create_corpus(self, sentences_clustered):
        docs_processed = self.init_nd_dict()
        for c in sentences_clustered.keys():
            for i, doc in enumerate(sentences_clustered[c]):
                processed = self.stem_data(self.remove_stopwords(word_tokenize(self.remove_punctuations(doc))))
                docs_processed[c][i] = processed

        return docs_processed

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

    def remove_punctuations(self, string):
        punctuations = [',','.', '?', ';',':','-','_',')','(',']','[','`','~','"','<','>','"']
        for p in punctuations:
            string = string.replace(p,'')
        return string

    def remove_stopwords(self, words):
        with open('stopwords.txt', 'r') as f:
            stopwords = f.read().split('\n')
        # temp.remove('')
        words_nostop = []
        # word_list = self.str_to_words(words)
        [words_nostop.append(x.lower()) for x in words if x not in stopwords]
        return words_nostop

    def stem_data(self, words_preprocessed):
        ps = PorterStemmer()
        words_stemmed = []
        for word in words_preprocessed:
            words_stemmed.append(ps.stem(word))
        return words_stemmed

    # calculate mutual information given all 4 counts
    def calc_mi(self, corpus, term, cls):
        N11, N10, N01, N00 = self.get_Ns(corpus, term, cls)

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
    def get_Ns(self, corpus, term, cls):
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

    def run_calculation(self, corpus, salt):
        result = self.init_nd_dict()
        start = time.time()
        for i, cls in enumerate(corpus.keys()):
            for doc in corpus[cls]:
                log.info('class: {}/{}---------------------------------------------------'.format(i+1, self.num_classes))
                log.info('calculating mutual information...{}/{} | running time: {} seconds'.format(doc, len(corpus[cls].keys()), round(time.time()-start,4)))
                for word in corpus[cls][doc]:
                    score = self.calc_mi(corpus, word, cls)
                    result[word][cls] = score

        with open('{}.json'.format('mi_scores_'+salt), 'w') as f:
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

    def sort_result(self, mi_result_path, save_name_salt):
        with open(mi_result_path, 'r') as f:
            mi_result = json.load(f)

        to_sort = self.init_nd_dict()
        for word in mi_result.keys():
            for corpus in mi_result[word]:
                score = mi_result[word][corpus]
                to_sort[corpus][word] = score

        result = self.init_nd_dict()

        for i in range(self.num_classes):
            result[i] = self.format_ranked_result(self.sort_dict_by_value(to_sort[str(i)]))

        with open('ranked_result_cluster_{}.json'.format(save_name_salt), 'w') as f:
            json.dump(result, f)

    def validate_result(self, ranked_cluster_path, num_keywords=10, num_max_sentences=5):
        superwords = self.init_nd_dict()

        with open(ranked_cluster_path) as f:
            results = json.load(f)

        # choose n "strongest" words and store them in a dict
        for i in list(results.keys()):
            for j in range(num_keywords):
                superwords[i][j] = list(results[str(i)].keys())[j]
        log.info(superwords)

        with open('sentences_clustered.json', 'r') as f:
            clusters = json.load(f)

        super_sentences = self.init_nd_dict()

        for cluster in clusters.keys():                     #for each cluster
            sentences_in_this_cluster = clusters[cluster]

            for s in sentences_in_this_cluster:
                for superword in superwords[cluster].values():
                    if superword in s and s not in super_sentences[cluster][superword]:   # if each superword is in any of the sentences in the same cluster
                        try:
                            super_sentences[cluster][superword].append(s)  # save the list of sentences in a dict
                        except:
                            super_sentences[cluster][superword] = [s]  # save the list of sentences in a dict

        self.format_supersentences(super_sentences, num_max_sentences)  #format the "supersentences" to be human-readable to a file

    def format_supersentences(self, super_sentences, max_sentences):
        with open('supersentences_per_cluster.txt', 'w') as f:
            for cluster in super_sentences.keys():
                f.write('cluster {}:\n'.format(cluster))

                for word in super_sentences[cluster].keys():
                    f.write('\t{}:'.format(word) + '\n')
                    for j, s in enumerate(super_sentences[cluster][word]):
                        if j == max_sentences:
                            break
                        f.write('\t\t{}\n'.format(s))

    def rank_words(self, documents):
        clusters_ranked = {}
        for i, doc in enumerate(documents):
            words_ranked = defaultdict(int)
            words_uniq = list(set(doc))
            for word in words_uniq:
                for j in range(len(doc)):
                    if word == doc[j]:
                        words_ranked[word] += 1
            clusters_ranked[i] = {k: float(v) for k, v in sorted(words_ranked.items(), key=lambda item: item[1], reverse=True)}

        return clusters_ranked

class UtilityFunct():

    def format_sentences(self, sentences, labels):
        splitted = []
        labels_extended = []    # each element in "sentences" is a set of multiple setences: therefore when they are split,
                                # the labels should be duplicated accordingly

        for i, s in enumerate(sentences):
            #remove unsupported punctuations
            sentence_chunk = s.replace('-',' ').replace('!','').replace('[',' ').replace(']',' ').replace(':',' ').replace(';',' or ')\
                            .replace('(a)',' ').replace('(b)','').replace('(c)','').replace('(d)','').replace('(e)','') \
                            .replace('(','').replace(')','').replace('\n','.')\
                            .replace('\u2019',' ').replace('\u2013',' ').replace('\u2014',' ').replace('\u201d',' ')\
                            .replace('\u201c',' ').replace('\u2018', ' ').replace('\u202f', ' ').replace('\u00e0', ' ')\
                            .replace('\u00e9',' ').replace('\u00a0', ' ').replace('(', ' ').replace(')',' ').replace('_',' ')\
                            .replace('   ',' ').replace('  ',' ').replace('    ',' ').replace('U.S.','USA').replace('E.U.','eu').replace('e.g.','for example,')\
                            .replace('(Japan)','japan')\
                            .replace('. ','.').replace('.','. ')\
                            .split('. ')

            splitted.append(sentence_chunk)

            for j in range(len(sentence_chunk)):
                labels_extended.append(labels[i])


        splitted = list(itertools.chain.from_iterable(splitted))

        formatted = []
        for sentence in splitted:
            if '.' not in sentence:
                formatted.append(str(sentence)+'.')
            else:
                formatted.append(str(sentence))

        log.info('[individual] sentences length: {}\n[individual] labels length: {}'.format(len(formatted),
                                                                                            len(labels_extended)))
        return formatted, labels_extended

#pre-determined centroid clustering
class PDC():

    def __init__(self):
        self.model = SentenceTransformer('snt_tsfm_model/')
        self.key_topics_titles = [
            'what data do we collect?',
            'How do we collect your data?',
            'How will we use your data?',
            'How do we store your data?',
            'Marketing',
            'What are your data protection rights?',
            'What are cookies?',
            'How do we use cookies?',
            'What types of cookies do we use?',
            'How to manage your cookies',
            'Privacy policies of other websites',
            'Changes to our privacy policy',
            'How to contact us',
            'How to contact the appropriate authorities'
        ]
        self.key_topics = [
        'we collect personal identification information such as name, email, phone number, etc and other necessary data.',

        'you directly provide us most of the data we collect your data when you register online, place order, voluntarily '
        'complete survey, provide feedback, use or view via cookies.',

        'we use your data to process order and manage account email you with special offers, share your data with partner '
        'companies, and send your data to credit reference agencies to prevent fraud and abuse.',

        'we securely retain, and maintain your data at until once this period time expired we delete'
        ' your data by months years.',

        'we send you information about products and services you might like recommend marketing third party use opt out '
        'later right to stop no longer wish marketing purposes.',

        'your data protection rights you have right to access rectify edit erase remove delete restrict processing object'
        ' data portable control transfer.',

        'what are cookies cookies are text files placed on your computer when you visit our website we collect through cookies.',

        'we use your cookies to keep you signed in understand how you use our website.',

        'we use different types of cookies, functionality remember your preferences language location advertising links you followed'
        'share online data with third parties for advertising authentication security performance analytics research.',

        'manage cookies, you can set your browser not to accept cookies remove cookies some of features not function as a result',

        'we contain links to other websites our privacy policy apply only to our website, if you click link to another website '
        'you should read and refer to their policy',

        'we keep our privacy policy under review and change regularly this was last updated on',

        'how to contact us if you have questions on privacy policy data we hold on you data about data protection rights',

        'how to contact the appropriate authorities and data protection officer report complaint information commissioner office'
    ]

    def get_predefined_centroids(self):
        return self.model.encode(self.key_topics)

    def get_n_best_from_predefined_centroids(self, sentences, n_best=5):
        key_topics_encoded = self.model.encode(self.key_topics)
        sentences_encoded = self.model.encode(sentences)

        cluster_result = defaultdict(dict)

        # dictinary that generates string ids for each sentence vector
        id2vector = dict(zip(list(map(self.generate_unique_id_from_sentence_vector, sentences_encoded)), sentences_encoded))
        id2sent = dict(zip(list(map(self.generate_unique_id_from_sentence_vector, sentences_encoded)), sentences))

        for i, t in enumerate(key_topics_encoded):
            cluster_id = str(i)
            for s in sentences_encoded:
                distance = euclidean(t,s)
                if len(cluster_result[cluster_id]) <= n_best:
                    sentence_id = self.generate_unique_id_from_sentence_vector(s)
                    cluster_result[cluster_id][sentence_id] = distance

                if len(cluster_result[cluster_id]) > n_best:
                    # sort items by distance
                    cluster_result[cluster_id] = {k: v for k, v in sorted(cluster_result[cluster_id].items(), key=lambda item: item[1])}

                    #remove last item (because its distance is the largest
                    cluster_result[cluster_id].popitem()

        #convert back vectors to human-readable sentences
        for cluster in cluster_result.keys():
            for sid in cluster_result[cluster].keys():
                cluster_result[cluster][sid] = {'text':id2sent[sid],
                                                'distance':cluster_result[cluster][sid]}

            cluster_result[cluster] = {'topic':self.key_topics_titles[int(cluster)],
                                       'members':cluster_result[cluster]}
        return cluster_result

    def generate_unique_id_from_sentence_vector(self, vector):
        product = 1
        for i in range(10):
            product *= vector[i]
        return str(product).split('e')[0][-8:]

    def run_clustering_with_predetermined_centroids(self, sentences, n_best=10):
        key_topics_encoded = self.model.encode(self.key_topics)
        sentences_encoded = self.model.encode(sentences)

        # dictionary that converts sentence id to vector
        id2vector = dict(zip(list(map(self.generate_unique_id_from_sentence_vector, sentences_encoded)), sentences_encoded))

        # find distance from each centroid to all sentences
        distances_n_best = defaultdict(dict)

        for i, t in enumerate(key_topics_encoded):
            cluster_id = str(i)
            for s in sentences_encoded:
                distance = euclidean(t,s)

                if len(distances_n_best[cluster_id]) <= n_best:
                    sentence_id = self.generate_unique_id_from_sentence_vector(s)
                    distances_n_best[cluster_id][sentence_id] = distance

                if len(distances_n_best[cluster_id]) > n_best:
                    # sort items by distance
                    distances_n_best[cluster_id] = {k: v for k, v in
                                                  sorted(distances_n_best[cluster_id].items(), key=lambda item: item[1])}

                    # remove last item (because its distance is the largest
                    distances_n_best[cluster_id].popitem()

        # for each sentence (sid) find distance to each centroid
        clustered = defaultdict(dict)
        for cid in distances_n_best.keys():
            for sentence_id in distances_n_best[cid].keys():
                distance = distances_n_best[cid][sentence_id]
                clustered[sentence_id][cid] = distance

        # sort by distance to find the closest centroid from each sentence (sid)
        for sentence_id in clustered.keys():
            # sort result by distance
            clustered[sentence_id] = {k: v for k, v in sorted(clustered[sentence_id].items(), key=lambda item: item[1])}
            # remove all others except for the closest one
            clustered[sentence_id] = next(iter(clustered[sentence_id]))

        # return cluster allocation result and the original sentence vectors
        return clustered, np.asarray(list(map(id2vector.get,list(clustered.keys()))))

class PPReporter():

    def is_title(self, sentence):
        lower_count = 0
        for char in sentence.replace(' ',''):
            if char.islower():
                lower_count += 1
        return lower_count == 0

    def generate_report(self, mode, url, n_best=1):
        # get Selenium work to get all the text from url and split by \n
        options = Options()
        driver = webdriver.Firefox(options=options)
        driver.get(url)
        time.sleep(2.5)
        body = driver.find_element(By.XPATH, '/html/body').text
        clauses = body.split('\n')
        driver.close()

        clauses_splitted = []
        for c in clauses:
            clauses_splitted.append([e+'.' for e in c.split('. ') if e])   #split by '. ' and add the full stop back to the sentence

        clauses_splitted = list(itertools.chain.from_iterable(clauses_splitted))

        # remove empty clauses
        try:
            clauses_splitted.remove('')
        except:
            pass

        # remove titles
        clauses_no_titles = []
        [clauses_no_titles.append(c) for c in clauses_splitted if not self.is_title(c)]
        log.info('original clauses length: %d' % len(clauses_no_titles))

        if mode == 'pdc':

            pdc = PDC()
            report = pdc.get_n_best_from_predefined_centroids(sentences=clauses_no_titles,
                                                              n_best=n_best)

            # format the result and write out as a report
            with open('pp_report_pdc.txt', 'w') as f:
                for cluster in report.keys():
                    f.write('[cluster {}]\n\n'.format(cluster))
                    f.write('\ttopic: {}\n\n'.format(report[cluster]['topic']))
                    for sid in report[cluster]['members'].keys():
                        f.write('\t\tclause found: "{}"\n'.format(report[cluster]['members'][sid]['text']))
                        f.write('\t\trelevance: {}\n\n'.format(round(report[cluster]['members'][sid]['distance'],6)))
                f.write('[END OF REPORT]\n')

            # return a list of raw clauses for later evaluation purposes
            raw_text = []
            for cluster in report.keys():
                for sid in report[cluster]['members'].keys():
                    raw_text.append(report[cluster]['members'][sid]['text'])

        elif mode == 'kmeans':
            with open('cluster_centers.json', 'r') as f:
                trained_centroids = json.load(f)

            tsfm = SentenceTransformer('snt_tsfm_model')
            raw_text = []
            centroids = list(itertools.chain.from_iterable(list(trained_centroids.values())))
            centroids_encoded = tsfm.encode(centroids)

            clauses_no_titles_encoded = tsfm.encode(clauses_no_titles)

            for i, c in enumerate(centroids_encoded):
                distances = [(euclidean(c, v), s) for s, v in zip(clauses_no_titles, clauses_no_titles_encoded)]    # calculate distance and keep the original sentence
                distances = sorted(distances)   #sort sentences by distance to trained centroids
                raw_text.append(distances[0][1])    #append closest sentence to retuning text
            # # only return the sentences & flatten the list because it's 2d (2d arrays are not accepted by SentenceTransformer
            # return list(itertools.chain.from_iterable(list(centers_dict.values())))

        else:
            raise ValueError('entered wrong mode! it should be either pdc or kmeans.')

        return raw_text

    # take a list of unencoded sentences and do evaluation on the summary report
    def evaluate_report(self, report_sentences):
        pdc = PDC()
        titles = pdc.model.encode(pdc.key_topics_titles)
        rs_encoded = pdc.model.encode(report_sentences)

        ssd = 0
        for t in titles:
            t_distance = 1e+10
            for rs in rs_encoded:
                d = euclidean(t,rs)
                if d < t_distance:
                    t_distance = d
            ssd += t_distance ** 2  # add squared minimal distance

        # log.info('report score: {}'.format(round(ssd,2)))
        return ssd

class RougeEvaluation():

    def __init__(self):
        self.rouge_model = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            # metrics=['rouge-n'],
            max_n=2,
            limit_length=True,
            length_limit=100,
            length_limit_type='words',
            apply_avg='Avg',
            apply_best='Best',
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True
        )

    def evaluate(self, hypothesis, reference):
        scores = self.rouge_model.get_scores(hypothesis=[hypothesis],
                                    references=[reference])
        # print(scores)
        return scores

def take_input(n_best, url):
    ppr = PPReporter()
    topics = [
            'what data do we collect?',
            'How do we collect your data?',
            'How will we use your data?',
            'How do we store your data?',
            'Marketing',
            'What are your data protection rights?',
            'What are cookies?',
            'How do we use cookies?',
            'What types of cookies do we use?',
            'How to manage your cookies',
            'Privacy policies of other websites',
            'Changes to our privacy policy',
            'How to contact us',
            'How to contact the appropriate authorities'
        ]
    raw_text_pdc = ppr.generate_report(url=url,
                                       mode='pdc',
                                       n_best=n_best)
    result = defaultdict(list)
    for i, t in enumerate(topics):
        result[t].append(raw_text_pdc[n_best * i : n_best * (i+1)])

    for key in result:
        temp = result[key]
        result[key] = list(itertools.chain.from_iterable(temp))

    return result

def evaluate_clauses(target_url):

    pdc = PDC()
    ppr = PPReporter()

    raw_text_pdc = ppr.generate_report(url=target_url,
                                       mode='pdc',
                                       n_best=3)

    raw_text_kms = ppr.generate_report(url=target_url,
                                       mode='kmeans')

    # direct sample from gdpr for benchmarking
    raw_text_gdpr = pdc.key_topics

    score_pdc = ppr.evaluate_report(raw_text_pdc)
    score_kms = ppr.evaluate_report(raw_text_kms)
    score_gdpr = ppr.evaluate_report(raw_text_gdpr)

    # try evaluation on randomly generated sentences for benchmarking
    with open('random_sentences.txt', 'r') as f:
        random_sentences = f.read().split('\n')

    # choose 14 from list of random sentences
    random_14 = random.sample(random_sentences, 14)
    score_rand = ppr.evaluate_report(random_14)

    target_url = target_url.replace('https://','').replace('www.','').replace('/','.')
    company_name = target_url.split('.')[0] + target_url.split('.')[1]
    return "%s,%.4f,%.4f,%.4f,%.4f" % (company_name, score_rand, score_pdc, score_kms, score_gdpr)

def calc_avg(list):
    sum = 0
    for item in list:
        sum += float(item)
    return sum / len(list)

def std_dev(list, avg):
    var = 0
    for item in list:
        var += (item-avg) ** 2

    return math.sqrt(var/len(list))


def record_evaluation_results_ssd():
    eval = pd.read_csv('evaluate_urls.csv')

    with open('ssd_evaluation.txt', 'w') as f:
        f.write(str(eval))
        score_rand = calc_avg(eval['score_rand'])
        rand_std = std_dev(eval['score_rand'], score_rand)

        score_pdc = calc_avg(eval['score_pdc'])
        pdc_std = std_dev(eval['score_pdc'], score_pdc)

        score_kms = calc_avg(eval['score_kms'])
        kms_std = std_dev(eval['score_kms'], score_kms)

        score_gdpr = calc_avg(eval['score_gdpr'])
        gdpr_std = std_dev(eval['score_gdpr'], score_gdpr)

        f.write('\n[SSD] random: %.4f | pdc: %.4f | kms: %.4f | GDPR: %.4f' % (score_rand,
                                                                             score_pdc,
                                                                             score_kms,
                                                                             score_gdpr))
        f.write('\n[std] random: %.4f | pdc: %.4f | kms: %.4f | GDPR: %.4f' % (rand_std,
                                                                            pdc_std,
                                                                            kms_std,
                                                                            gdpr_std))

def visualise_evaluation_results_ssd():
    eval = pd.read_csv('evaluate_urls.csv')
    companies = eval['company_name']
    score_rand = eval['score_rand']
    score_pdc = eval['score_pdc']
    score_kms = eval['score_kms']
    score_gdpr = eval['score_gdpr']

    fig, ax = plt.subplots()
    ax.plot(companies, score_rand, label='random')
    ax.plot(companies, score_pdc, label='PDC')
    ax.plot(companies, score_kms, label='K-means')
    ax.plot(companies, score_gdpr, label='GDPR', linestyle='-', color='blue')
    ax.set_xticks(companies[::2])
    ax.set_xticklabels(companies[::2], rotation=75)
    ax.set_ylabel('Sum of Squared Distance')
    ax.set_xlabel('Privacy Policy company name')
    plt.legend(bbox_to_anchor=(0.99,1.0), loc='upper left')
    plt.savefig('ssd_plot.png', bbox_inches='tight')

def main():

    """""""download sentence transformer model and save"""""""
    # model = SentenceTransformer('stsb-mpnet-base-v2')
    # model.save('snt_tsfm_model/')
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""init knn module and encode clauses"""""""""""""""""""""""""
    clu = Clustering()
    ut = UtilityFunct()
    sentences_raw = pd.read_csv('training_data/alice/data_alice.csv')['clause']
    labels_raw = pd.read_csv('training_data/alice/data_alice.csv')['class']
    sentences, labels = ut.format_sentences(sentences_raw, labels_raw)
    # sum = 0
    # for s in sentences:
    #     sum += len(s)
    # print('%d words on average!'%(sum/len(sentences)))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""do Pre-Deteremined centroid clustering"""""""""""""""""""""""""""""
    # pdc = PDC()
    # result = pdc.get_n_best_from_predefined_centroids(sentences,10)
    # for cluster in result.keys():
    #     log.info('topic: {}'.format(result[cluster]['topic']))
    #     for member in result[cluster]['members'].keys():
    #         log.info('{}: {}'.format(member, result[cluster]['members'][member]))
    #     log.info('\n')

    # pdc_clustered_dict, sentence_vectors = pdc.run_clustering_with_predetermined_centroids(sentences, n_best=100000)
    # pca_vectors, _ = clu.do_pca(n_components=3, sentences=sentence_vectors)
    # labels = list(map(int, list(pdc_clustered_dict.values())))
    # clu.plot_labels(pca_vectors, labels, 'pdc') #plot clustering results
    # sil = silhouette_score(pca_vectors, labels)
    # log.info(sil)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""do embedding and load embedded sentences to be processed"""""""""
    # clu.do_sentenceBERT(sentences, labels)
    # with open('stnce_embedding.json', 'r') as f:
    #     sent_dict = json.load(f)
    #
    # embedded_sentences = list(sent_dict.values())
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""do pca on embedded sentences"""""""""""""""""""""""""""""""""""""""""""""
    # clu.find_best_ncomp_for_pca(embedded_sentences)
    # pca_sentences, pca2vector = clu.do_pca(n_components=10,sentences=embedded_sentences)   #90% var: 140, 85%:100 80%: 80 75% 65 70%: 45

    ### find best ncomp by silhouette scores
    # sil = []
    # k = 14
    # for i, n in enumerate(range(3,200)):
    #     log.info('doing silhouette scores on various ncomp ... {}/{}'.format(i, len(list(range(3,200)))))
    #     pca_sentences, pca2vector = clu.do_pca(n_components=n,
    #                                            sentences=embedded_sentences)  #90% var: 140, 85%:100 80%: 80 75% 65 70%: 45
    #     sil.append(clu.do_silhouette_single(pca_sentences, k))
    #
    # y = sil
    # x = range(3,200)
    # plt.plot(x, y)
    # plt.xlabel('Number of Components')
    # plt.ylabel('Silhouette score')
    # plt.savefig('silhouette_by_ncomps_k={}.png'.format(k))
    # plt.show()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""perform elbow method / silhouette method to find optimal k for kmeans"""""
    # clu.do_elbow(pca_sentences,50)
    # clu.do_silhouette(pca_sentences,50)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""perform different clustering methods"""""""""""""""""""""""""""""""""""""""""
    # distance_metric = 'euclidean'
    # k = 14
    #
    # # run clustering and compare results using many clustering algorithms
    # algorithms = ['kmeans','kmeans_mini','dbscan', 'hdbscan','gaussian',
    #                'birch','affinity','meanshift','optics','agglomerative']
    # algorithms_competitive = ['kmeans','kmeans_mini','gaussian','birch', 'agglomerative']
    # clustering_result = {}
    # for al in algorithms:
    #      sil = clu.perform_clustering(pca_sentences,al, num_clusters=k, metric=distance_metric)
    #      clustering_result[al] = sil
    # log.info(clustering_result)

    # ### sort dictinoary by value
    # sorted_clustering_result = {k: float(v) for k, v in sorted(clustering_result.items(), key=lambda item: item[1],reverse=True)}
    #
    # for item in sorted_clustering_result.items():
    #     log.info(item)

    # ###do further analysis and report using kmeans because it gives the best silhouette score
    # clu.perform_clustering(pca_sentences, 'kmeans', num_clusters=k, metric=distance_metric, pca2vector=pca2vector)
    # with open('cluster_centers.json', 'r') as f:
    #     centers = json.load(f)
    # for item in centers.items():
    #     log.info(item)
    # clu.format_cluster_centres()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""" 'evaluate' k-means by printing out the n best sentences in each cluster """""""""""""""""
    # ec = EvalCluster(num_classes=14)
    # with open('sentences_clustered.json','r') as f:
    #     sc = json.load(f)
    # corpus = ec.create_corpus(sc)
    # ec.run_calculation(corpus=corpus, salt='original')
    # ec.sort_result('mi_scores_original.json', 'original')
    # ec.validate_result('ranked_result_cluster_original.json',
    #                    num_keywords=7,
    #                    num_max_sentences=1)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""generate report on PP (SSD) using a url & evaluate"""""""""""""""""""""""""""""""""
    # url = 'https://discord.com/privacy'
    # print(take_input(3,url))

    ### get list of urls to test
    # with open('pps_to_eval.txt', 'r') as f:
    #     urls_to_eval = f.read().split('\n')
    # try:
    #     urls_to_eval.remove('')
    # except:
    #     pass

    # ## run evaluation on all urls
    # with open('evaluate_urls.csv', 'w') as f:
    #     f.write('company_name,score_rand,score_pdc,score_kms,score_gdpr\n')
    #     for url in urls_to_eval:
    #         f.write(evaluate_clauses(url)+'\n')

    # play around with the SSD evaluation result
    # record_evaluation_results_ssd()
    # visualise_evaluation_results_ssd()

    ### test take input module for flask app
    # n_best = 1
    # url = 'https://stackoverflow.com/legal/privacy-policy'
    # take_input(n_best, url)

    """""""""""""""""""""""""""""""""generate report on PP (ROUGE) using a url & evaluate"""""""""""""""""""""""""""""""""
    target_url = 'https://twitter.com/en/privacy'
    ppr = PPReporter()
    rg = RougeEvaluation()

    #prepare human summary
    human_summaries = pd.read_csv('human_summary.csv')
    twitter = human_summaries['twitter']

    raw_text_pdc = ppr.generate_report(url=target_url,
                                       mode='pdc',
                                       n_best=3)

    raw_text_kms = ppr.generate_report(url=target_url,
                                       mode='kmeans')


    #run scoring for each topic
    for pdc_s, kms_s, human_s in zip(raw_text_pdc, raw_text_kms, twitter):
        pdc_score = rg.evaluate(hypothesis=pdc_s, reference=human_s)
        kms_score = rg.evaluate(hypothesis=kms_s, reference=human_s)
        print('pdc score: {}\nkms score: {}\n\n'.format(json.dumps(pdc_score), json.dumps(kms_score)))

if __name__ == '__main__':
    main()