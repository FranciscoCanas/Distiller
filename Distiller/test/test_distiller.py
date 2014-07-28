from dircache import listdir
import os
import unittest
from Distiller.distiller import Distiller

test_data = {
        'metadata': {
            'base_url': 'http://github.com/franciscocanas/Distiller/'
        },
        'documents': [
            {
                'id': 1,
                'body': "Far far away, behind the word mountains, far from the countries "
                        "Vokalia and Consonantia, there live the blind texts. Separated they "
                        "live in Bookmarksgrove right at the coast of the Semantics, a large "
                        "language ocean. A small river named Duden flows by their place and "
                        "supplies it with the necessary regelialia. It is a paradisematic country, "
                        "in which roasted parts of sentences fly into your mouth. Even the "
                        "all-powerful Pointing has no control about the blind texts it is an "
                        "almost unorthographic life One day however a small line of blind text "
                        "by the name of Lorem Ipsum decided to leave for the far World of Grammar. "
                        "The Big Oxmox advised her not to do so, because there were thousands of "
                        "bad Commas, wild Question Marks and devious Semikoli, but the Little Blind Text did not "
                        "listen. She packed her seven versalia, put her initial into the belt and made herself "
                        "on the way."
            },
            {
                'id': 2,
                'body': "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed "
                        "in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted "
                        "his head a little he could see his brown belly, slightly domed and divided by arches "
                        "into stiff sections. The bedding was hardly able to cover it and seemed ready to "
                        "slide off any moment. His many legs, pitifully thin compared with the size of the "
                        "rest of him, waved about helplessly as he looked. \"What's happened to me?\\"
            },
            {
                'id': 3,
                'body': "A wonderful serenity has taken possession of my entire soul, like these sweet "
                        "mornings of spring which I enjoy with my whole heart. I am alone, and feel the "
                        "charm of existence in this spot, which was created for the bliss of souls like "
                        "mine. I am so happy, my dear friend, so absorbed in the exquisite sense of mere "
                        "tranquil existence, that I neglect my talents. I should be incapable of drawing "
                        "a single stroke at the present moment; and yet I feel that I never was a greater "
                        "artist than now. When, while the lovely valley teems with vapour around me, and the "
                        "meridian sun strikes the upper surface of the impenetrable foliage of my trees, "
                        "and but a few stray gleams steal into the inner sanctuary, I throw myself down "
                        "among the tall grass by the trickling stream"
            }
        ]
    }

data = 'data/data.json'
result = 'data/tests/'

nlp_args = {
    'normalize': True,
    'stem': False,
    'lemmatize': False,
    'tfidf_cutoff': 0.0001,
    'pos_list': ['NN','NP', 'JJ'],
    'black_list': []
}

def clean_folder(path):
    """
    Remove any existing files or folders at path.
    """
    if os.path.exists(path):
        for fileName in listdir(path):
            os.remove(path + '/' + fileName)
        os.rmdir(path)

class TestDistiller(unittest.TestCase):
    """
    Basic testing for Distiller class.
    """

    def test_DistillerOutputFiles(self):
        """
        Runs Distiller on test data and checks that output files are produced.
        """
        clean_folder(result)
        Distiller(data, result, nlp_args, verbosity=3)
        self.assertTrue(os.path.exists(result + 'keymap.json'))
        self.assertTrue(os.path.exists(result + 'docmap.json'))
        self.assertTrue(os.path.exists(result + 'keywords.json'))
        self.assertTrue(os.path.exists(result + 'bigrams.json'))
        self.assertTrue(os.path.exists(result + 'trigrams.json'))

    def test_DistillerDocs(self):
        """
        Runs on test data and checks that all stats are collected.
        """
        clean_folder(result)
        d = Distiller(data, result, nlp_args, verbosity=3)
        self.assertTrue(d.processed_documents)
        self.assertTrue(d.statistics)
        self.assertTrue(d.statistics['keywords'])
        self.assertTrue(d.statistics['bigrams'])
        self.assertTrue(d.statistics['trigrams'])





