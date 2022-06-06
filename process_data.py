import sys, os.path
import pandas as pd
from sqlalchemy import create_engine


messages_filepath = "./messages.csv", 
categories_filepath = "./categories.csv"
database_filepath = './DisasterResponse_processed_data.db'

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df
    pass

def clean_data(df):
    categories = df["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[0]
    new_row = []
    for item in row:
        item = item[:-2]
        new_row.append(item)
    category_colnames = new_row
    categories.columns = category_colnames
    for column in categories:      
        categories[column] = categories[column].str.strip().str[-1] # set each value to be the last character of the string
    
        categories[column] = pd.to_numeric(categories[column]) # convert column from string to numeric
    #categories.head()
    df = df.drop(["categories"], axis=1) # drop the original categories column from `df`

    df = pd.concat([df, categories], axis=1, join='inner') # concatenate the original dataframe with the new `categories` dataframe
    
    df.drop_duplicates() # drop duplicates
    
    df = df.replace(2,1) #related-2 doesn't make sense, so we replcae it witha one
    return df
    pass

def save_data(df, database_filepath):
    if os.path.isfile(database_filepath):
        print("File already exists")

    else:
        engine = create_engine('sqlite://'+ database_filepath[1:])
        df.to_sql('InsertTableName', engine, index=False)
    pass  

def main():
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data("./messages.csv", "./categories.csv")

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
        
    print('Cleaned data saved to database!')
    
if __name__ == '__main__':
    main()