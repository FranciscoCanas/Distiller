Distiller
=========

Distiller provides convenient auto-extraction of document key words
based on term-frequency/inverse-document-frequency (TF-IDF) and word
positioning.

Distiller handles all of the pre-processing details and produces final
statistic reports in JSON format.


Requirements
------------

Distiller uses the [Natural Language Toolkit](http://www.nltk.org/)

You will need to download a couple of NLTK packages:

    >>> import nltk
    >>> nltk.download()
    Downloader> d
    Download which package (l=list; x=cancel)?
        Identifier> maxent_treebank_pos_tagger
    Downloader> d
    Download which package (l=list; x=cancel)?
        Identifier> stopwords



Installation
------------

Installation using pip:

    $ pip install Distiller


Usage
-----

Typical usage from within the Python interpreter:

    >>> from Distiller.distiller import Distiller
    >>> distiller = Distiller(data, target, options)


Arguments
---------

### data

Path to file containing the document collection in JSON format.

    {
        'metadata': {
            'base_url': 'The document's source URL (if any)'
            },
        'documents': [
                {
                    'id': 'The document's unique identifier (if any)',
                    'body': 'The entire body of the document in a single text blob.',
                }, ...
            ]
    }

###target

Path where Distiller will output the following reports:

keywords: A list of words and the frequency with which they were detected as being keywords of documents.

bigrams: A list of word pairs and the frequency with which they were detected as being key pairs in documents.

trigrams: A list of word triples and the frequency with which they were detected as being key pairs in documents.

docmap: A mapping of document IDs to their respective keywords, n-grams, and other statistics.

keymap: A mapping of keywords to the documents they appear in.


###options

An optional dictionary containing document processing arguments in this format:

    {
        'normalize': True,          # normalize tokens during pre processing
        'stem': True,               # stems tokens during pre processing
        'lemmatize': False,         # lemmatize during pre processing
        'tfidf_cutoff': 0.001,      # cutoff value to use for term-freq/doc-freq score
        'pos_list': ['NN','NNP'],   # POS white list used to filter for candidates
        'black_list': []            # token list used to filter out from candidates
    }

