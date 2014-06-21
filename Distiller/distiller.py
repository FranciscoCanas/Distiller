import os
import json

import nltk

from features.Collocations import Collocations
from features.Positioning import Positioning
from features.tf_idf import tf_idf
from preprocessing.pipeline import pre_process_pipeline


__author__ = 'fcanas'

class Distiller():
    """
    Responsible for extracting features from a set of documents. Expects documents specified in JSON format:

    {
    metadata : {
        base_url : "..."
        }
    documents : [{
        id: INT,
        body: "...",
        }, ...],
    }


    The documents will be processed and kept in processed_documents dic, which will have this format:

    {
    id => {
            id : INT,

            tokenized_body : [token1, token2, ...] # where tokens are lowerecase unicode words that form text body

            # the resulting list from calling pre_process_pipeline()
            processed_body : [(processed_token1, POS1), (processed_token2, POS2),...],

            candidates: [candidate1, candidate2, ...], # likely candidate words are determined by pipeline

            freq_distribution:  {token1: frequency2, token2: frequency2} # frequencies for all tokens in body
        }
    }

    nlp_args:

    A dictionary of arguments used by the pre-processing pipeline. Takes the following form:

    {
        normalize: Boolean,                 # normalize tokens during pre processing
        stem: Boolean,                      # stems tokens during pre processing
        lemmatize: Boolean,                 # lemmatize during pre processing
        tfidf_cutoff: Float,                # cutoff value to use for term-freq/doc-freq score
        pos_list: [STRING,...],             # POS white list used to filter for candidates
        black_list: [token1, token2, ...]   # token list used to filter out from candidates
    }

    """

    default_args = {
        'normalize': True,          # normalize tokens during pre processing
        'stem': True,               # stems tokens during pre processing
        'lemmatize': False,         # lemmatize during pre processing
        'tfidf_cutoff': 0.001,      # cutoff value to use for term-freq/doc-freq score
        'pos_list': ['NN','NNP'],   # POS white list used to filter for candidates
        'black_list': []            # token list used to filter out from candidates
    }

    def __init__(self, document_file, target_path, nlp_args=default_args):
        """
        Initialize Distiller for specified document file.
        """
        self.processed_documents = {}
        self.statistics = {}
        self.tfidf = tf_idf()
        self.collocations = Collocations()
        self.positioning = Positioning()
        self.path = make_path(target_path)

        with open(document_file) as d:
            self.jdata = json.load(d)
        d.close()

        self.initialize_arguments(nlp_args)
        self.process_documents()
        self.compute_statistics()
        self.export()

    def initialize_arguments(self, nlp_args):
        """
        Read args from json metadata and initialize Distiller arguments.
        """
        self.base_url = self.jdata['metadata']['base_url']
        self.documents = self.jdata['documents']
        self.normalize = nlp_args['normalize']
        self.stem = nlp_args['stem']
        self.lemmatize = nlp_args['lemmatize']
        self.pos_list = nlp_args['pos_list']
        self.tfidf_cutoff = nlp_args['tfidf_cutoff']
        self.black_list = nlp_args['black_list']

    def process_documents(self):
        """
        Run documents from json through pre-processing.
        """
        for document in self.documents:
            doc = {}
            doc['id'] = document['id']
            doc['url'] = self.base_url.format(int(document['id']))
            doc['tokenized_body'] = map(unicode.lower, nltk.word_tokenize(document['body']))
            doc['processed_tokens'] = pre_process_pipeline(text=document['body'],
                                                     black_list=self.black_list,
                                                     pos_list=self.pos_list,
                                                     normalize=self.normalize,
                                                     stem=self.stem,
                                                     lemmatize=self.lemmatize)
            doc['candidates'] = list(set(zip(*doc['processed_tokens'])[0]))
            doc['freq_distribution'] = nltk.FreqDist(doc['tokenized_body'])
            self.processed_documents[doc['id']] = doc

        self.processed_doc_bodies = [document['processed_tokens'] for document in self.processed_documents.values()]

    def compute_statistics(self):
        """
        Compute the statistics for each pre-processed document, given the entire body of docs.
        """
        for document in self.processed_documents.values():
            document['tfidf'] = self.tfidf.compute_tf_idf(document['candidates'],
                                                           document['tokenized_body'],
                                                           self.processed_doc_bodies)

            document['positioning'] = self.positioning.compute_position_score(document['candidates'],
                                                                              document['tokenized_body'])

            document['keywords'] = self.extract_keywords(document['tfidf'],
                                                         document['positioning'],
                                                         lower_cutoff=self.tfidf_cutoff)

            document['bigrams'] = self.collocations.find_ngrams(2, document['processed_tokens'])

            document['trigrams'] = self.collocations.find_ngrams(2, document['processed_tokens'])



    def extract_keywords(self, tf_idf_scores, positioning_scores, lower_cutoff=0.0001):
        """
        Input: sorted list of tf-idf candidate scores
               hash of position scores.

        output: list of (candidate, score=tf-idf * positioning)
        """
        keywords = []
        for candidate in tf_idf_scores:
            if candidate[1] > lower_cutoff:
                score = candidate[1] * 2 * positioning_scores[candidate[0]]
                keywords.append((candidate[0], score))
        return keywords

    def export(self):
        """
        Write all of the stats and processed documents out to the target path.
        """
        export_statistic(self.path, self.processed_documents, 'keywords', lambda x: x[0], nltk.FreqDist)
        export_statistic(self.path, self.processed_documents, 'bigrams', lambda x: ' '.join(map(lambda y: y[0], x)), nltk.FreqDist)
        export_statistic(self.path, self.processed_documents, 'trigrams', lambda x: ' '.join(map(lambda y: y[0], x)), nltk.FreqDist)
        store_collections(self.path, self.processed_documents)


def export_statistic(path, docs, stat, transformer=lambda x: x, compiler=lambda x: x):
    """
    Creates a json output file for the given stat and set of bugs.
    """
    stats = []
    for doc in docs.values():
        for item in doc[stat]:
            stats.append(transformer(item))
    processed_statistic = compiler(stats)
    with open(path + '/' + stat + '.json', 'w') as outfile:
        json.dump(processed_statistic, outfile)
    outfile.close()



def store_collections(path, documents):
    """
    Stores the collections:
    (keyword => [BZ id, ...])
    (BZ id => Bug)
    """
    keymap = {}
    docmap = {}

    for doc in documents.values():
        docmap[doc['id']] = doc
        for word in doc['keywords']:
            if not keymap.has_key(word[0]):
                keymap[word[0]] = []
            keymap[word[0]].append(str(doc['id']))

    with open(path + '/keymap.json', 'w') as outfile:
        json.dump(keymap, outfile)
    outfile.close()

    with open(path + '/docmap.json', 'w') as outfile:
        json.dump(docmap, outfile)
    outfile.close()

def make_path(path):
    """
    Ensures the target path for stat reports is created.
    """
    if not path.endswith('/'):
        path = path + '/'

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    return path
