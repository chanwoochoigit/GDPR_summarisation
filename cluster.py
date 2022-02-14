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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch, \
                            AffinityPropagation, MeanShift, OPTICS, \
                            AgglomerativeClustering
import hdbscan
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

        #create sentence embedding dict[plain_text] = sentence_vector
        with open('stnce_embedding.json', 'w') as f:
            json.dump(sentence_dict, f)

        #create reversed sentence embedding dict[sentence_vector] = plain_text
        with open('stnce_embedding_reverse.json', 'w') as f:
            json.dump(sentence_dict_reverse, f)

    def cluster_decode(self, X, pred, num_clusters):
        sent_cluster_dict = {}
        with open('stnce_embedding_reverse.json', 'r') as f:
            sent_dict_reverse = json.load(f)

        for x, p in zip(X, pred):   # convert sentence vector to plain text, and put it in a dictionary
                                    # dict[plain_text] = cluster
            try:
                sent_cluster_dict[str(sent_dict_reverse[str(x.tolist())])] = str(p)
                # print(str(sent_dict_reverse[str(x.tolist())])+': '+str(p))
            except AttributeError:  # decoding list of centroids
                sent_cluster_dict[str(sent_dict_reverse[str(x)])] = str(p)

        #decode encoded sentences and save to a dict -> dict[cluster_num] = decoded_vector
        cluster_dict = {}
        for i in range(num_clusters):
            print('saving decoded sentence vectors ... {}/{}'.format(i, num_clusters))
            temp = []
            for k, v in sent_cluster_dict.items():
                if str(v) == str(i):
                    temp.append(k)
            cluster_dict[i] = temp

        return cluster_dict

    def generate_pca_key(self, principle_comp):
        # create pseudo-unique keys to create kv table
        return str(principle_comp[0]/principle_comp[1]*principle_comp[2])[3:]

    # return nearest point in a dataset
    def find_nearest_point(self, data, point):
        distance = 1e+10
        nearest = point
        for d in data:
            temp_dist = euclidean(d, point)
            if temp_dist < distance:
                print('from {} to {}'.format(distance, temp_dist))
                distance = temp_dist
                nearest = d

        return nearest

    # def find_centers_kmeans(self, pca2vector, kmeans, sentences, num_clusters):
    #     # collect centers and find the closest vector to get approximate center vector
    #     # this process is required because a center vector is a mean of other vectors and not be able to be decoded
    #     centers = kmeans.cluster_centers_
    #     centers_decodable = {}
    #     start = time.time()
    #     for i, center in enumerate(centers):
    #         print('finding appx. center for cluster {}/{} ... running time: {} seconds'.format(i, num_clusters, round(
    #             time.time() - start, 4)))
    #         center = pca2vector[self.generate_pca_key(center)]
    #         distance = 1e+10  # base distance: to be updated so a random large number is given
    #         for sent in sentences:
    #             temp_dist = euclidean(center, sent)
    #             if temp_dist < distance:
    #                 print('from {} to {}'.format(distance, temp_dist))
    #                 distance = temp_dist
    #                 centers_decodable[i] = sent
    #
    #     #predicted clusters for centers are just cluster numbers in order, so just put list(0 - 20) as pred
    #     return self.cluster_decode(list(centers_decodable.values()),list(range(num_clusters)),num_clusters)

    def plot_labels(self, df, pred, algorithm):
        u_labels = np.unique(pred)
        for i in u_labels:
            plt.scatter(df[pred == i, 0], df[pred == i, 1], label=i)
        plt.legend()
        plt.savefig('clustering_results/{}.png'.format(algorithm))

    # perform unsupervised clustering and compare results for different clustering algorithms
    def perform_clustering(self, sentences, clusterer, metric, num_clusters=None, pca2vector=None):
        # prepare data
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2') # normalise vector to make cosine similarity effect
        labels = []
        sil = 999

        if clusterer == 'kmeans':
            if num_clusters is not None:
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(normed_X)
                labels = kmeans.predict(normed_X)

                if pca2vector is not None:

                    # get centroids and nearest points for further analysis

                    centers = kmeans.cluster_centers_
                    pseudo_centers = []
                    for c in centers:  # get nearest point from each center
                        pseudo_c = self.find_nearest_point(sentences, c)
                        pseudo_centers.append(pca2vector[self.generate_pca_key(pseudo_c)])
                    centers_decoded = self.cluster_decode(pseudo_centers, list(range(10)), num_clusters)
                    with open('cluster_centers.json', 'w') as f:
                        json.dump(centers_decoded, f)

                    pca_decoded = []
                    for pca in X:
                        pca_decoded.append(pca2vector[self.generate_pca_key(pca)])
                    # decode clustered sentences to plain text
                    cluster_dict = self.cluster_decode(pca_decoded, labels, num_clusters)
                    with open('sentences_clustered.json', 'w') as f:
                        json.dump(cluster_dict, f)

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
            raise ValueError('chosen clusterer not supported')

        print(len(set(labels))) # get number of clusters for algorithms not requiring num_cluster param
        try:
            sil = silhouette_score(normed_X, labels)
            print(sil)
        except ValueError:
            print('clustering failed miserably! There is only a single cluster.')

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
            print('finding the optimum cluster number ... {}/{} | running time: {} seconds'.format(k,kmax,round(time.time()-start,4)))
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

    def find_best_ncomp_for_pca(self, sentences):
        pca = PCA()
        pca_sentences = pca.fit_transform(sentences)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
        plt.xlabel('Components')
        plt.ylabel('Explained Variances')
        plt.show()

    def do_pca(self, n_components,sentences):

        pca = PCA(n_components=n_components, random_state=42)
        pca_sentences = pca.fit_transform(sentences)

        #create a dictionary that converts principle components to sentence vectors for later
        pca2vector = {}
        for pca, vector in zip(pca_sentences, sentences):
            pca2vector[self.generate_pca_key(pca)] = vector

        return pca_sentences, pca2vector

    def do_silhoulette(self, sentences, kmax):
        X = np.asarray(sentences)
        normed_X = normalize(X, axis=1, norm='l2')

        sil = []
        start = time.time()
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            print('finding the optimum cluster number ... {}/{} | running time: {} seconds'.format(k,kmax,round(time.time()-start,4)))
            kmeans = KMeans(n_clusters=k).fit(normed_X)
            labels = kmeans.labels_
            sil.append(silhouette_score(normed_X, labels, metric='euclidean'))

        y = sil
        x = range(len(y))
        plt.plot(x, y)
        plt.savefig('silhouette.png')
        plt.show()

class EvalCluster():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    def create_corpus(self):
        with open('sentences_clustered.json', 'r') as f:
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

    def run_calculation(self):
        corpus = self.load_corpus()
        result = self.init_nd_dict()
        start = time.time()
        for i, cls in enumerate(corpus.keys()):
            for doc in corpus[cls]:
                print('class: {}/{}---------------------------------------------------'.format(i+1, self.num_classes))
                print('calculating mutual information...{}/{} | running time: {} seconds'.format(doc, len(corpus[cls].keys()), round(time.time()-start,4)))
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

    def sort_result(self):
        with open('mi_scores.json', 'r') as f:
            to_display = json.load(f)
        to_sort = self.init_nd_dict()
        for word in to_display.keys():
            for corpus in to_display[word]:
                score = to_display[word][corpus]
                to_sort[corpus][word] = score

        result = self.init_nd_dict()

        for i in range(self.num_classes):
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

        with open('sentences_clustered.json', 'r') as f:
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

def main():
    """""""""""init knn module and encode clauses"""""""""""
    clu = Clustering()
    ut = UtilityFunct()
    # with open('clauses_v2.pkl', 'rb') as f:
    #     sentences = pickle.load(f)
    # clu.do_sentenceBERT(ut.split_sentences(sentences))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""load embedded sentences to be processed"""""""""
    with open('stnce_embedding.json', 'r') as f:
        sent_dict = json.load(f)

    embedded_sentences = list(sent_dict.values())
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""do pca on embedded sentences"""""""""""""""""""""""""""""""""""""""""""""
    # clu.find_best_ncomp_for_pca(embedded_sentences)
    pca_sentences, pca2vector = clu.do_pca(n_components=3,sentences=embedded_sentences)   #90% var: 160, 85%:120 80%: 90 75% 70 70%: 56

    """""perform elbow method / silhouette method to find optimal k for kmeans"""""
    # clu.do_elbow(pca_sentences,50)
    # clu.do_silhoulette(pca_sentences,50)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""perform different clustering methods"""""""""""""""""""""""""""""""""""""""""
    distance_metric = 'euclidean'
    k = 10
    algorithms = ['kmeans','kmeans_mini','dbscan', 'hdbscan','gaussian',
                  'birch','affinity','meanshift','optics','agglomerative']
    algorithms_competitive = ['kmeans','kmeans_mini','gaussian','birch', 'agglomerative']
    clustering_result = {}
    # for al in algorithms_competitive:
    #     sil = clu.perform_clustering(pca_sentences,al, num_clusters=k, metric=distance_metric)
    #     clustering_result[al] = sil
    #
    # sorted_clustering_result = {k: float(v) for k, v in sorted(clustering_result.items(), key=lambda item: item[1],reverse=True)}
    #
    # for item in sorted_clustering_result.items():
    #     print(item)

    ##go further with kmeans cos it's the best atm
    clu.perform_clustering(pca_sentences, 'kmeans', num_clusters=k, metric=distance_metric, pca2vector=pca2vector)
    with open('cluster_centers.json', 'r') as f:
        centers = json.load(f)
    for item in centers.items():
        print(item)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""evaluate k-means"""""""""""""""""
    # eval = EvalCluster(num_classes=10)
    # eval.create_corpus()
    # eval.run_calculation()
    # eval.sort_result()
    # eval.validate_result(num_keywords=5, max_sentences=5)
    """"""""""""""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':
    main()