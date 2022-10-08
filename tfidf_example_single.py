#!/usr/bin/env python
#
# Created in 2022 by Dogan Yigit Yenigun (toUpperCase78)
# (yigit_yenigun@hotmail.com)
#
# Example usage of methods defined in TF-IDF library for single document.

__mod_author__ = "Dogan Yigit Yenigun"
__mod_email__ = "yigit_yenigun at hotmail dot com"

import math
import re
import codecs
import tfidf
from operator import itemgetter

file = codecs.open('EnglishText/english_text1.txt', 'r')
txtdata = ""
for line in file:
    txtdata += line.replace('\n', '') + ' '
# print("TEXT DATA:", txtdata)
file.close()

my_tfidf = tfidf.TfIdf(stopword_filename='english_stopwords.txt',
                       DEFAULT_IDF = 1.0)
tokens = my_tfidf.get_tokens(txtdata)
print("TOKENS:", tokens)
keywords = my_tfidf.get_doc_keywords(txtdata)
print("KEYWORDS & TFIDF:", keywords)

my_tfidf.add_input_document(txtdata)
my_tfidf.get_tfidf_status()

my_tfidf.save_corpus_to_file('idf_single.txt', 'non_stopwords_single.txt')
