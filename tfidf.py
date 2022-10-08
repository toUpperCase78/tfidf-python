#!/usr/bin/env python
# 
# Copyright 2009  Niniane Wang (niniane@gmail.com)
# Reviewed by Alex Mendes da Costa.
#
# Modified in 2012 by Benjamin Fields (me@benfields.net)
#
# Further reviewed and modified in 2022 by Dogan Yigit Yenigun (toUpperCase78)
# (yigit_yenigun@hotmail.com)
#
# This is a simple TF-IDF library.  The algorithm is described in:
#   https://en.wikipedia.org/wiki/Tf-idf
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# TF-IDF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details:
#   https://www.gnu.org/licenses/lgpl.txt

__orig_author__ = "Niniane Wang"
__orig_email__ = "niniane at gmail dot com"
__mod_author__ = "Dogan Yigit Yenigun"
__mod_email__ = "yigit_yenigun at hotmail dot com"

import math
import re
import codecs
from operator import itemgetter

class TfIdf:
    """
        TF-IDF class implementing https://en.wikipedia.org/wiki/Tf-idf.
        The library constructs an IDF corpus and stopword list either from
        documents specified by the client, or by reading from input files.
        It computes IDF for a specified term based on the corpus, or generates
        keywords ordered by TF-IDF for a specified document.
    """
    def __init__(self, corpus_filename = None, stopword_filename = None,
                 DEFAULT_IDF = 1.5):
        """
        Initialize the IDF dictionary.
       
        If a corpus file is supplied, reads the IDF dictionary from it,
        in the format of:
            # of total documents (1st line)
            term : # of documents containing the term (2nd and subsequent lines)

        If a stopword file is specified, reads the stopword list from it, in
        the format of one stopword per line.

        The DEFAULT_IDF value is returned when a query term is not found in the
        IDF corpus.
        """
        self.num_docs = 0
        self.term_num_docs = {}     # term : num_of_docs_containing_term
        self.stopwords = set([])
        self.idf_default = DEFAULT_IDF

        if corpus_filename:
            self.merge_corpus_document(corpus_filename)

        if stopword_filename:
            stopword_file = codecs.open(stopword_filename, "r", encoding='utf-8')
            self.stopwords = set([line.strip() for line in stopword_file])
            stopword_file.close()

    def get_tfidf_status(self):
        """Show the contents of all attributes for the TFIDF object."""
        print("# OF DOCUMENTS:", self.num_docs)
        print("# OF TERMS IN DOCUMENTS:", self.term_num_docs)
        print("STOPWORDS:", self.stopwords)
        print("IDF DEFAULT VALUE:", self.idf_default)

    def get_tokens(self, str):
        """
        Break a string into tokens, preserving URL tags as an entire token.
        This implementation does not preserve case.  
        Clients may wish to override this behavior with their own tokenization.
        """
        return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())

    def merge_corpus_document(self, corpus_filename):
        """Take the corpus document, and add it to the existing corpus model."""
        corpus_file = codecs.open(corpus_filename, "r", encoding='utf-8')

        # Load number of documents.
        line = corpus_file.readline()
        self.num_docs += int(line.strip())

        # Reads "term:frequency" from each subsequent line in the file.
        for line in corpus_file:
            tokens = line.rsplit(":",1)
            term = tokens[0].strip()
            try:
                frequency = int(tokens[1].strip())
            except(IndexError, err):
                if line in ("","\t"):
                    # Catch blank lines
                    print("# Line is blank #")
                    continue
                else:
                    raise
            if term in self.term_num_docs.keys():
                self.term_num_docs[term] += frequency
            else:
                self.term_num_docs[term] = frequency
        corpus_file.close()

    def add_input_document(self, input):
        """Add terms in the specified document to the IDF dictionary."""
        self.num_docs += 1
        words = set(self.get_tokens(input))
        for word in words:
            if word not in self.stopwords:
                if word in self.term_num_docs:
                    self.term_num_docs[word] += 1
                else:
                    self.term_num_docs[word] = 1

    def save_corpus_to_file(self, corpus_filename, non_stopwords_filename,
                            STOPWORD_PERCENTAGE_THRESHOLD = 0.01):
        """Save the IDF dictionary and non-stopword list to the specified files."""
        output_file = codecs.open(corpus_filename, "w", encoding='utf-8')

        output_file.write(str(self.num_docs) + "\n")
        for term, num_docs in self.term_num_docs.items():
            output_file.write(term + ":" + str(num_docs) + "\n")

        sorted_terms = sorted(self.term_num_docs.items(), key=itemgetter(1), reverse=True)
        non_stopwords_file = open(non_stopwords_filename, "w")
        for term, num_docs in sorted_terms:
            if num_docs < STOPWORD_PERCENTAGE_THRESHOLD * self.num_docs:
                break
            non_stopwords_file.write(term + "\n")
        print("The corpus was saved successfully...")

    def get_num_docs(self):
        """Return the total number of documents in the IDF corpus."""
        return self.num_docs

    def get_idf(self, term):
        """
        Retrieve the IDF for the specified term. 
        This is computed by taking the logarithm of
        (1 + # of documents in corpus) / (1 + # of documents containing this term)
        """
        if term in self.stopwords:
            return 0

        if not term in self.term_num_docs:
            return self.idf_default

        return math.log(float(1 + self.get_num_docs()) / (1 + self.term_num_docs[term]))

    def get_doc_keywords(self, curr_doc):
        """
        Retrieve terms and corresponding TF-IDF values for the specified single document.
        The returned terms are ordered by decreasing TF-IDF.
        """
        tfidf = {}
        tokens = self.get_tokens(curr_doc)
        tokens_set = set(tokens)
        for word in tokens_set:
            tf = float(tokens.count(word)) / len(tokens_set)
            idf = self.get_idf(word)
            if idf != 0.0:
                tfidf[word] = tf * idf

        return sorted(tfidf.items(), key=itemgetter(1), reverse=True) 
