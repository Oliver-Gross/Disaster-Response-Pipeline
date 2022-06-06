import json
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import sqlalchemy as sq
import joblib
from sqlalchemy import create_engine
import train_classifier
   
app = Flask(__name__)

database_filepath = './DisasterResponse_processed_data.db'

def tokenize(text):
    '''
    INPUT:  (message)                   -   a string
    OUTPUT: (clean_tokens)              -   a numpy.ndarray with the tokenized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite://'+ database_filepath[1:])
df = pd.read_sql_table('InsertTableName', engine)
z = df[df.columns.drop(['id','message', 'original','genre', 'related', 'request', 'offer'])]

# load model
model = pickle.load(open("./model.pkl", 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    amount_categories = df[df.columns.drop(['id','message', 'original','genre'])].sum(axis=0, skipna=True)
    amount_categories = amount_categories.sort_values(ascending=False)
    
    #data for second visual
    top_5 = z.sum().sort_values(ascending=False).head()
    top_names = list(top_5.index)
    top_categories = top_5.tolist()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    #x=genre_names,
                    #y=genre_counts
                    x = top_names,
                    y = top_categories
                )
            ],

            'layout': {
                'title': 'Distribution of top 5 categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
     }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()