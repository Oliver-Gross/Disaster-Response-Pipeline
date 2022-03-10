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

database_filepath = './DisasterResponse_processed_data.db'
model_filepath = 'model.pkl'

def load_data(database_filepath):
    engine = sq.create_engine('sqlite://'+ database_filepath[1:])
    df = pd.read_sql_table('InsertTableName', engine)
    X = df.message.values
    y = df[df.columns.drop(['id','message', 'original','genre'])]
    return X, y
    pass

def tokenize(message):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, message)
    for url in detected_urls:
        message = message.replace(url, "urlplaceholder")
    tokens = word_tokenize(message)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass

def build_model():
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10, random_state=1)))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
    }
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    #print(pipeline.get_params())
    #return cv
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    prediction = model.predict(X_test)
    converted_list = [str(element) for element in y_test.iloc[0].tolist()]
    converted_list2 = [str(element) for element in prediction[0]]
    print('New-Y-List:',converted_list)
    print('New-Pred:',converted_list2)
    print('Recall:',recall_score(converted_list, converted_list2, average='weighted'))
    print('F1-Score:',f1_score(converted_list, converted_list2, average='weighted'))
    print('Precision:',precision_score(converted_list, converted_list2, average='weighted'))
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb')) #'model.pkl'
    pass

def main():
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
    print('Building model...')
    model = build_model()
        
    print('Training model...')
    model.fit(X_train, Y_train)
        
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

if __name__ == '__main__':
    main()