import pymongo
import yaml

from pymongo import UpdateOne
from tqdm import tqdm
from urllib.parse import quote_plus

class MongoConnection:
    def __init__(self, **kwargs):
        self.host = kwargs.pop('host', None)
        self.port = kwargs.pop('port', None)
        self.user_name = kwargs.pop('user_name', None)
        self.password = kwargs.pop('password', None)
        self.auth_db = kwargs.pop('auth_db', None)
        self.uri = kwargs.pop('uri', None)

    def __repr__(self):
        return (
            f"==={self.__class__.__qualname__}===\n" + 
            f"HOST : {self.host}\n" +
            f"PORT : {self.port}\n" +
            f"USER : {self.user_name}"
        )


    def get_client(self):
        if self.uri is None:
            self.uri = "mongodb://%s:%s@%s/%s" % (
                quote_plus(self.user_name),
                quote_plus(self.password), 
                self.host,
                self.auth_db,
            )

        return pymongo.MongoClient(self.uri)

    def get_detail(id, db_name):
        db = mongo_client['example']
        collection = db[db_name]
        if type(id) is not int:
            c_num = int(class_number)
        document = collection.find({"_id" : class_number})
        document = list(document)
        return document[0]

    def upsert(self, data, collection, batch_size=None):
        def batch(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]  

        if None in [data, collection]:
            print("Data and collection name should not be empty")
            return
            
        if type(data) is not list and type(data[0]) is not dict:
            print("Data should be of type list and its elements should be of type dictionary")
            return
        
        if batch_size is None:
            batch_size = len(data)
        
        batched_operations = []
        for subset in batch(data, batch_size):
            batched_operations.append(subset)

        results = []
        for _, batch in enumerate(tqdm(batched_operations)):
            results.append(collection.bulk_write(batch))

        return results
        
    def make_operation(self, field, _filter=None, upsert=True):
        return UpdateOne(
            _filter if _filter is not None else {},
            {"$set": field},
            upsert=upsert
        )      
