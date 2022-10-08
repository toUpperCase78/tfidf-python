#!/usr/bin/env python
#
# Created in 2022 by Dogan Yigit Yenigun (toUpperCase78)
# (yigit_yenigun@hotmail.com)
#
# Example usage of methods defined in TF-IDF library for multiple documents.

__mod_author__ = "Dogan Yigit Yenigun"
__mod_email__ = "yigit_yenigun at hotmail dot com"

import math
import re
import codecs
import tfidf
import os
from operator import itemgetter

my_tfidf = tfidf.TfIdf(stopword_filename='english_stopwords.txt',
                       DEFAULT_IDF = 1.0)

num_text_files = len(os.listdir(os.getcwd()+'/EnglishText'))

for i in range(1,num_text_files+1):
    file = codecs.open(os.getcwd()+'/EnglishText/english_text'+str(i)+'.txt', 'r')
    txtdata = ""
    for line in file:
        txtdata += line.replace('\n', '') + ' '
    file.close()
    # print("TEXT DATA:", txtdata)
    print("TEXT", i, "->", my_tfidf.get_doc_keywords(txtdata)[:20])
    my_tfidf.add_input_document(txtdata)

my_tfidf.get_tfidf_status()
print("There are total of", my_tfidf.get_num_docs(), "documents in this TF-IDF.")

my_tfidf.save_corpus_to_file('idf_multi.txt', 'non_stopwords_multi.txt')
