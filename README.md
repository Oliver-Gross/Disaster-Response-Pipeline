# Disaster-Response-Pipeline
## Project for Udacity Data Scientist Nanodegree
![](https://upload.wikimedia.org/wikipedia/commons/3/3b/Udacity_logo.png)

### Table of Contents

1. [Installations for process_data.py](#insta_pro)
2. [Description for process_data.py](#desc_pro)
3. [Installations for train_classifier.py](#insta_train)
4. [Description for train_classifier.py](#desc_train)
5. [Installations for run.py](#insta_run)
6. [Description for run.py](#desc_run)
7. [Instructions](#instructions)
8. [Authors and Acknowledgements](#licensing)

## process_data.py <a name="insta_pro"></a>
## Installations

import pandas as pd <br>
from sqlalchemy import create_engine

[Panda](https://pandas.pydata.org/) is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. <br>
[create engine](https://docs.sqlalchemy.org/en/14/core/engines.html) creates an Engine instance.

## Description <a name="desc_pro"></a>
The first part of the data pipeline is the extract, transform, and load process. Here the dataset will be read, cleaned and then stored in a SQLite database.  

## train_classifier.py <a name="insta_train"></a>
## Installations

import sys, nltk, re <br>
import sqlalchemy as sq <br>
import pandas as pd <br>
import numpy as np <br>
import pickle <br>
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger']) <br>

from nltk.tokenize import word_tokenize <br>
from nltk.stem import WordNetLemmatizer <br>
from sklearn.pipeline import Pipeline <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer <br>
from sklearn.multioutput import MultiOutputClassifier <br>
from sklearn.ensemble import RandomForestClassifier <br>
from sklearn.model_selection import GridSearchCV <br>
from sklearn.metrics import f1_score, precision_score, recall_score <br>

[sys](https://docs.python.org/3/library/sys.html) provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. <br>
[nltk](https://www.nltk.org/) provides easy-to-use interfaces, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries. <br>
[re](https://docs.python.org/3/library/re.html) provides regular expression matching operations similar to those found in Perl. <br>
[sqlalchemy](https://www.sqlalchemy.org/) is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL. <br>
[numpy](https://numpy.org/) is a fundamental package for scientific computing with Python . <br>
[pickle](https://docs.python.org/3/library/pickle.html) implements binary protocols for serializing and de-serializing a Python object structure. <br>
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) applies a list of transforms and a final estimator. <br>
[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) splits arrays or matrices into random train and test subsets. <br>
[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) converts a collection of text documents to a matrix of token counts. <br>
[TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) transforms a count matrix to a normalized tf or tf-idf representation. <br>
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) is a simple strategy for extending classifiers that do not natively support multi-target classification. <br>
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. <br>
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) searchs over specified parameter values for an estimator. <br>
[sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) quantifies the quality of predictions. <br>

## Description <a name="desc_train"></a>
In the second part of the data pipeline we will split the data into a training set and a test set. Then, we will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict for 36 categories (multi-output classification). At the end we export
the model to a pickle file.

## run.py <a name="insta_run"></a>
## Installations

import json <br>

from flask import Flask <br>
from flask import render_template, request, jsonify <br>
import joblib <br>
from sqlalchemy import create_engine <br>

[json](https://www.w3schools.com/python/python_json.asp) is a syntax for storing and exchanging data.. <br>
[Flask](https://www.fullstackpython.com/flask.html) is a Python web framework built with a small core and easy-to-extend philosophy. <br>
[render_template](https://www.fullstackpython.com/flask-templating-render-template-examples.html) is used to generate output from a template file based on the Jinja2 engine that is found in the application's templates folder. <br>
[request](https://docs.python-requests.org/en/latest/) is an elegant and simple HTTP library for Python. <br>
[jsonify](https://www.fullstackpython.com/flask-json-jsonify-examples.html) is a function in Flask's flask.json module. jsonify serializes data to JavaScript Object Notation (JSON) format, wraps it in a Response object with the application/json mimetype. <br>
[joblib](https://joblib.readthedocs.io/en/latest/) is a set of tools to provide lightweight pipelining in Python. <br>

## Description <a name="desc_run"></a>
The last part we'll display your results in a Flask web app. [Udacity](https://www.udacity.com/) provided a working web app.
On the web app ther is an input fieled for distress messages, which will be analysed after pushing the "Classify Message" button.
Below we see the distribution of the message genres.
The analysis shows how the message has been categorized.

## Instructions <a name="instructions"></a>
1. Run the following commands to set up your database and model.

2. Go to `app` directory: `cd app`

3.    - To run ETL pipeline that cleans data and stores in database <br>
        `python process_data.py`
      - To run ML pipeline that trains classifier and saves <br>
        `python train_classifier.py`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage
## Authors, Acknowledgements <a name="licensing"></a>

Author: Oliver Groß
