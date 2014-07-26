from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re
import nltk

__author__ = 'fcanas'

class Pipeline():
    """
    Responsible for pre-processing any given document by performing the following set of steps:
    1) Divide content into tokens.
    2) POS tag tokens.
    3) Filter out only potential candidates.
    3) Normalize
    4) Filter out useless tokens.
    5) Stem.
    6) Lemmatize
    """

    def __init__(self,pos_tag=True, black_list=[], pos_list=['NN', 'JJ', 'NNP'], normalize=True, stem=True, lemmatize=True):
        self.pos_tag = pos_tag
        self.black_list = black_list
        self.pos_list=pos_list
        self.normalize = normalize
        self.stem = stem
        self.lemmatize = lemmatize


    def pre_process(self, text):
        """
        Input: "Blob of text..."
        Output: [(token, tag), ...]
        """
        processed = []
        tokens = self.tokenize(text)
        if self.pos_tag: tokens = self.pos_tag_tokens(tokens)

        for token in tokens:
            if not self.filter_by_pos(token) or not self.filter_by_pattern(token):
                continue
            if self.normalize: token = self.normalize_text(token)
            if self.stem: token = self.stem_token(token)
            if self.lemmatize: token = self.lemmatize_token(token)
            processed.append(token)
        return processed


    def tokenize(self, text):
        """
        Input: "Body of text...:
        Output: [word, ...] list of tokenized words matching regex '\w+'
        """
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        return tokens


    def pos_tag_tokens(self, tokens):
        """
        Use nltk's pos taggers on the given tokens.
        Input: [token1, token2, ...]
        Output: [(token1, tag1), (token2, tag2), ...]
        """
        return nltk.pos_tag(tokens)


    def filter_by_pos(self, token):
        """
        Input: (word, pos)
        Output: True if pos is noun, proper noun, or adj. False otherwise.
        """
        return token[1] in self.pos_list


    def normalize_text(self, token):
        """
        Input: (word, pos)
        Output: (word, pos) with lower case word.
        """
        return token[0].lower(), token[1]


    def stem_token(self, token):
        """
        Input: (word, pos)
        Output: (stem of word, pos)
        """
        porter = nltk.PorterStemmer()
        return porter.stem(token[0]), token[1]


    def lemmatize_token(self, token):
        """
        Input: (word, pos)
        Output: (Lemmatized word, pos)
        """
        wnl = nltk.WordNetLemmatizer()
        return wnl.lemmatize(token[0]), token[1]


    def filter_by_pattern(self, token):
        """
        Filters by removing tokens with words that are numbers, or in the black list or in the stop words list.

        Input: (word, pos), [blacklist_word, ...]
        Output: False if word is any of the above. True otherwise.
        """
        nums = re.compile("^[0-9]*$")
        return token[0].lower() not in map(unicode, self.black_list) and \
               token[0].lower() not in stopwords.words('english') and \
               not nums.match(token[0])
