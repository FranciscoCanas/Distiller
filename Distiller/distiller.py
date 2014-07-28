import os
import json
import logging
import nltk

from features.Collocations import Collocations
from features.Positioning import Positioning
from features.tf_idf import TF_IDF
from preprocessing.pipeline import Pipeline



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

    def __init__(self, document_file, target_path, nlp_args=default_args, verbosity=2):
        """
        Initialize Distiller for specified document file.
        """
        self.processed_documents = {}
        self.statistics = {}
        self.collocations = Collocations()
        self.positioning = Positioning()
        self.path = make_path(target_path)
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=self.get_logging_level(verbosity))

        with open(document_file) as d:
            self.jdata = json.load(d)
        d.close()

        self.initialize_arguments(nlp_args)
        self.process_documents()
        self.tfidf = TF_IDF(self.processed_doc_bodies)
        self.extract_features()
        self.compile()
        self.export()

    def initialize_arguments(self, nlp_args):
        """
        Read args from json metadata and initialize Distiller arguments.
        """
        logging.info("initializing Distiller")
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
        pipeline = Pipeline(black_list=self.black_list,
                            pos_list=self.pos_list,
                            normalize=self.normalize,
                            stem=self.stem,
                            lemmatize=self.lemmatize)

        for document in self.documents:
            logging.info("processing document {0}".format(document['id']))
            doc = {}
            doc['id'] = document['id']
            doc['url'] = self.base_url.format(int(document['id']))
            doc['tokenized_body'] = map(unicode.lower, nltk.word_tokenize(document['body']))
            doc['processed_tokens'] = pipeline.pre_process(text=document['body'])
            if not doc['processed_tokens']:
                doc['candidates'] = []
            else:
                doc['candidates'] = list(set(zip(*doc['processed_tokens'])[0]))
            doc['freq_distribution'] = nltk.FreqDist(doc['tokenized_body'])
            self.processed_documents[doc['id']] = doc

        self.processed_doc_bodies = [document['processed_tokens'] for document in self.processed_documents.values()]

    def extract_features(self):
        """
        Extract the features for each pre-processed document, given the entire body of docs.
        """
        for document in self.processed_documents.values():
            logging.info("computing statistics for {0}".format(document['id']))
            document['tfidf'] = self.tfidf.compute(document['candidates'],
                                                           document['tokenized_body'])

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

    def compile(self):
        """
        Compile the statistics of all extracted features.
        """
        self.compile_statistic('keywords', lambda x: x[0], nltk.FreqDist)
        self.compile_statistic('bigrams', lambda x: ' '.join(map(lambda y: y[0], x)), nltk.FreqDist)
        self.compile_statistic('trigrams', lambda x: ' '.join(map(lambda y: y[0], x)), nltk.FreqDist)
        self.compile_collections()

    def export(self):
        """
        Write all of the stats and processed documents out to the target path.
        """
        logging.info("exporting statistics to {0}".format(self.path))
        for key, value in self.statistics.items():
            export_to_file(self.path, key, value)

        export_to_file(self.path, 'keymap', self.keymap)
        export_to_file(self.path, 'docmap', self.docmap)

    def compile_statistic(self, stat, transformer=lambda x: x, compiler=lambda x: x):
        """
        Creates a json output file for the given stat and set of bugs.
        """
        stats = []
        for doc in self.processed_documents.values():
            for item in doc[stat]:
                stats.append(transformer(item))
        self.statistics[stat] = compiler(stats)

    def compile_collections(self):
        """
        Stores the collections:
        (keyword => [BZ id, ...])
        (BZ id => Bug)
        """
        self.keymap = {}
        self.docmap = {}
        logging.info("storing document and keyword collections to {0}".format(self.path))
        for doc in self.processed_documents.values():
            self.docmap[doc['id']] = doc
            for word in doc['keywords']:
                if not self.keymap.has_key(word[0]):
                    self.keymap[word[0]] = []
                self.keymap[word[0]].append(str(doc['id']))

    def get_logging_level(self, verbosity):
        """
        Return a logging level based on verbosity argument {0,1,2}.
        """
        if verbosity < 1:
            return logging.ERROR
        if verbosity == 1:
            return logging.WARNING
        if verbosity == 2:
            return logging.INFO
        if verbosity > 2:
            return logging.DEBUG



def export_to_file(path, stat, col):
    """
    Export the given stat to file in the given path.
    """
    with open(path + '/' + stat + '.json', 'w') as outfile:
        json.dump(col, outfile)
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
