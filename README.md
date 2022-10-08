# tfidf-python

_A simple yet useful TF-IDF Python class_

## Details

This TF-IDF implementation performs the following below:

* Constructs an IDF corpus and stopword list either from documents specified by the user, or by reading from input files
* Displays the current status of TF-IDF
* Creates tokens based on the regular expression
* Computes IDF for a specified term based on the corpus
* Generates keywords ordered by TF-IDF for a specified document
* Saves the corpus and non-stopword files to the storage

Original work can be found here: http://code.google.com/p/tfidf/

This repo was forked from https://github.com/gearmonkey/tfidf-python. However, it was left untouched by the repo owner **for more than 10 years**! Therefore, the main purpose of this fork is to overhaul everything for usability and compatibility of the latest versions of Python (e.g. version 3.10).

In **EnglishText** directory, you can find 12 long texts written in English to better simulate the functionality of the TF-IDF implementation. All texts include news that correpond to Formula 1, technology and gaming, at a time from 2022.
