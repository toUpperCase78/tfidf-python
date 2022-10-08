#!/usr/bin/env python
# 
# Copyright (C) 2009.  All rights reserved.
#
# Further reviewed and modified by Dogan Yigit Yenigun (toUpperCase78) in 2022.

__orig_author__ = "Niniane Wang"
__orig_email__ = "niniane at gmail dot com"
__mod_author__ = "Dogan Yigit Yenigun"
__mod_email__ = "yigit_yenigun at hotmail dot com"

import math
import tfidf
import unittest

DEFAULT_IDF_UNITTEST = 1.0

def get_idf_value(num_docs_total, num_docs_term):
   return math.log(float(1 + num_docs_total) / (1 + num_docs_term))

def get_tfidf_status(tfidf):
   print("# of documents:", tfidf.num_docs)
   print("# of terms in documents:", tfidf.term_num_docs)
   print("Stopwords:", tfidf.stopwords)
   print("Default IDF Value:", tfidf.idf_default)

class TfIdfTest(unittest.TestCase):
   def testGetIdf(self):
      """Test querying the IDF for existent and nonexistent terms."""
      print("\nInitializing testGetIdf...")
      my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", \
                             DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
      get_tfidf_status(my_tfidf)

      # Test querying for a nonexistent term.
      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("nonexistent"))
      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("THE"))

      self.assertTrue(my_tfidf.get_idf("a") > my_tfidf.get_idf("the"))
      self.assertAlmostEqual(my_tfidf.get_idf("girl"), my_tfidf.get_idf("moon"))
      print("PASSED")

   def testKeywords(self):
      """Test retrieving keywords from a document, ordered by TF-IDF."""
      print("\nInitialzing testKeywords...")
      my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", DEFAULT_IDF = 0.01)
      get_tfidf_status(my_tfidf)

      # Test retrieving keywords when there is only one keyword.
      keywords = my_tfidf.get_doc_keywords("the spoon and the fork")
      print(keywords)
      self.assertEqual("the", keywords[0][0])

      # Test retrieving multiple keywords.
      keywords = my_tfidf.get_doc_keywords("the girl said hello over the phone")
      print(keywords)
      self.assertEqual("girl", keywords[0][0])
      self.assertEqual("phone", keywords[1][0])
      self.assertEqual("said", keywords[2][0])
      self.assertEqual("the", keywords[3][0])
      print("PASSED")

   def testAddCorpus(self):
      """Test adding input documents to the corpus."""
      print("Initializing testAddCorpus...")
      my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", \
                             DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
      get_tfidf_status(my_tfidf)

      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 1),
                             my_tfidf.get_idf("moon"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 5),
                             my_tfidf.get_idf("said"))

      my_tfidf.add_input_document("water, moon")

      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 1),
                             my_tfidf.get_idf("water"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 2),
                             my_tfidf.get_idf("moon"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 5),
                             my_tfidf.get_idf("said"))
      print("PASSED")

   def testNoCorpusFiles(self):
      print("\nInitializing testNoCorpusFiles...")
      my_tfidf = tfidf.TfIdf(DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
      get_tfidf_status(my_tfidf)

      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("moon"))
      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("said"))

      my_tfidf.add_input_document("moon")
      my_tfidf.add_input_document("moon said hello")

      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 1),
                             my_tfidf.get_idf("said"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 2),
                             my_tfidf.get_idf("moon"))
      print("PASSED")

   def testStopwordFile(self):
      print("\nInitializing testStopwordFile...")
      my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", "tfidf_teststopwords.txt",
                             DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
      get_tfidf_status(my_tfidf)

      self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
      self.assertEqual(0, my_tfidf.get_idf("moon"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 5),
                             my_tfidf.get_idf("said"))

      my_tfidf.add_input_document("moon")
      my_tfidf.add_input_document("moon and water")

      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 1),
                             my_tfidf.get_idf("water"))
      self.assertEqual(0, my_tfidf.get_idf("moon"))
      self.assertAlmostEqual(get_idf_value(my_tfidf.get_num_docs(), 5),
                             my_tfidf.get_idf("said"))
      print("PASSED")

# Need to add some UTF-8 handling tests

def main():
   unittest.main()

if __name__ == '__main__':
   main()
