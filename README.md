# Disaster-Response-Pipeline
## Project for Udacity Data Scientist Nanodegree
![](https://upload.wikimedia.org/wikipedia/commons/3/3b/Udacity_logo.png)

## process_data.py
## Installations

import pandas as pd
import sys, os.path
from sqlalchemy import create_engine

[Panda](https://pandas.pydata.org/) is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. <br>
[sys](https://docs.python.org/3/library/sys.html) provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. <br>
[os.path](https://docs.python.org/3/library/os.path.html) implements some useful functions on pathnames. <br>
[create engine](https://docs.sqlalchemy.org/en/14/core/engines.html) creates an Engine instance.

## train_classifier.py
## Installations

import sys, nltk, re
import sqlalchemy as sq
import pandas as pd
import numpy as np
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

[nltk](https://www.nltk.org/) provides easy-to-use interfaces,along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries. <br>
[re](https://docs.python.org/3/library/re.html) provides regular expression matching operations similar to those found in Perl. <br>
[sqlalchemy](https://www.sqlalchemy.org/) is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL. <br>
[numpy](https://numpy.org/) is a fundamental package for scientific computing with Python . <br>
[pickle](https://docs.python.org/3/library/pickle.html) implements binary protocols for serializing and de-serializing a Python object structure. <br>
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) applies a list of transforms and a final estimator.. <br>
[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) splits arrays or matrices into random train and test subsets. <br>
[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) converts a collection of text documents to a matrix of token counts. <br>
[TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) transforms a count matrix to a normalized tf or tf-idf representation. <br>
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) is a simple strategy for extending classifiers that do not natively support multi-target classification. <br>
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. <br>
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) searchs over specified parameter values for an estimator. <br>
[sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) quantifies the quality of predictions. <br>

## run.py
## Installations

import json

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine

[json](https://www.w3schools.com/python/python_json.asp) is a syntax for storing and exchanging data.. <br>
[Flask](https://www.fullstackpython.com/flask.html) is a Python web framework built with a small core and easy-to-extend philosophy. <br>
[render_template](https://www.fullstackpython.com/flask-templating-render-template-examples.html) is used to generate output from a template file based on the Jinja2 engine that is found in the application's templates folder. <br>
[request](https://docs.python-requests.org/en/latest/) is an elegant and simple HTTP library for Python. <br>
[jsonify](https://www.fullstackpython.com/flask-json-jsonify-examples.html) is a function in Flask's flask.json module. jsonify serializes data to JavaScript Object Notation (JSON) format, wraps it in a Response object with the application/json mimetype. <br>
[joblib](https://joblib.readthedocs.io/en/latest/) is a set of tools to provide lightweight pipelining in Python. <br>
