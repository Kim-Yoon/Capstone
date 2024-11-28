from motor.motor_asyncio import AsyncIOMotorClient
import yaml

env_path = './.env.yml'

with open(env_path) as f:
    env = yaml.load(f, Loader=yaml.FullLoader)
    
mongo_client = AsyncIOMotorClient(env['mongo']['uri'])
db = mongo_client['example']

class Finder:
    def __init__(self, collection_name):
        self.results = {}
        self.collection_name = collection_name
        self.collection = db[collection_name]

    async def search_in_field(self, field, keyword):
        documents = self.collection.find({field: {'$regex': keyword, '$options': 'i'}})
        await self.add_result(documents)

    async def search_all(self, keyword):
        query = { "$text": { "$search": keyword } }
        documents = self.collection.find(query)
        await self.add_result(documents)

    async def add_result(self, documents):
        async for item in documents:
            self.results[item['_id']] = item['name']

    async def send_result(self, keyword):
        await self.search_in_field('name', keyword)
        await self.search_in_field('location', keyword)
        await self.search_in_field('hashtag', keyword)
        await self.search_all(keyword)
        return self.results
        
        

    
    