import pymongo

client = pymongo.MongoClient('mongodb+srv://apple825:aa04190825@cluster0.amq3ff3.mongodb.net/?retryWrites=true&w=majority')

post = client['core_data']['post']

print(data)