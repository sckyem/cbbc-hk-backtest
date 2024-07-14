from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from default_modules import *
import streamlit as st

class Mongodb:

    def __init__(self, collection='test', document='test'):
        self.HOST = '127.0.0.1'
        self.PORT = 27017
        self.COLLECTION = collection
        self.DOCUMENT = document

    def atlas(self, is_ping=False):
        self.secret = st.secrets['mongodbpw']
        uri = f"mongodb+srv://{self.secret}@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        try:
            conn = MongoClient(uri, server_api=ServerApi('1'))
            if is_ping:
                conn.admin.command('ping')
                print("Pinged your deployment. You successfully connected to MongoDB!")
            return conn
        except Exception as e:
            print(e)

    def server(self, is_list_dbs=False):
            client = MongoClient(f"mongodb://{self.HOST}:{self.PORT}")
            if is_list_dbs:
                print(client.list_database_names())
            return client

    def document(self):
        conn = self.atlas()     
        return conn[self.COLLECTION][self.DOCUMENT]

    def insert(self, dataframe, batch_size=1000):
        df = dataframe.copy()
        if isinstance(dataframe, pd.Series):
            df = dataframe.to_frame() 

        if isinstance(df, pd.DataFrame):
            df.columns = columns_to_strings(df.columns)
            df['_id'] = df.index
            data = df.to_dict('records')
        else:
            data = df

        document = self.document()
        if isinstance(data, dict):
            result = document.insert_one(data)
            print("Inserting data Success" if result.acknowledged else "Error.")

        elif isinstance(data, list):
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                result = document.insert_many(batch)
                print("Inserting data Success" if result.acknowledged else "Error.")                
        else: print(f"{type(data)} cant insert" )

    def find(self, query={}, projection={}, is_dataframe=False):
        document = self.document()
        result = list(document.find(query, projection))
        if result:
            if is_dataframe:
                df = pd.DataFrame(result)
                df = df.set_index(df.columns[0])
                return df
            else:
                return result

    def update(self, data):
        new = data.to_frame() if isinstance(data, pd.Series) else data.copy()
        if isinstance(new, pd.DataFrame) and not new.empty:
            if isinstance(new.columns, pd.MultiIndex):
                new.columns = columns_to_strings(new.columns)
            old = self.find({'_id': {'$gte': new.index[0], '$lte': new.index[-1]}}, is_dataframe=True) 
            if isinstance(old, pd.DataFrame):
                if (set(old.columns) == set(new.columns)) and (type(old.index) == type(new.index)):
                    mask = new.ne(old).any(axis=1)                    
                    update_df = new.loc[new.index.isin(mask[mask].index)]
                    if not update_df.empty:
                        document = self.document() 
                        for index, row in update_df.iterrows():
                            filter = {'_id': index}
                            update = {'$set': row.to_dict()}
                            result = document.update_one(filter, update, upsert=True)
                        print(  f"Update " + f"{'success' if result.acknowledged else 'error'}" + f" for {self.COLLECTION} {self.DOCUMENT}"  )
                        return True
                    else:
                        print(  f"No update for {self.COLLECTION} {self.DOCUMENT}."  )
                        return False
                else:
                    print("The column names are different")
                    print(list(old.columns))
                    print(list(new.columns))
            else:
                if not self.find():
                    self.insert(new)
                    print("test")
                    return True
                else:
                    print(  f"Update error for {self.COLLECTION} {self.DOCUMENT}"  )

    def read(self, query={}, projection={}, is_dataframe=False):
        return self.find(query, projection, is_dataframe)

    def write(self, data):
        return self.update(data)
    
    def last_id(self):
        document = self.document()
        last_document = document.find_one(sort=[('_id', -1)])
        return last_document['_id']
    
    def first_id(self):
        document = self.document()
        return document.find_one()['_id']
    
if __name__ == '__main__':

    Mongodb('192.168.1.21', 27017).conn(is_list_dbs=True)