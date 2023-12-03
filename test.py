collection_1 = ['MATICUSDT', 'UNIUSDT', 'LTCUSDT', 'FILUSDT', 'AXSUSDT', 'SUIUSDT', 'XLMUSDT']
collection_2 = ['INJUSDT', 'GRTUSDT', 'DOGEUSDT', 'SNXUSDT', 'APTUSDT', 'GALAUSDT', 'SOLUSDT']
collection_3 = ['AVAXUSDT', 'STXUSDT', 'SANDUSDT', 'THETAUSDT', 'RNDRUSDT', 'NEOUSDT', 'APEUSDT']
collection_4 = ['XRPUSDT', 'ATOMUSDT', 'ALGOUSDT', 'DYDXUSDT', 'MANAUSDT', 'KAVAUSDT']

collection = []
collection.extend(collection_1)
collection.extend(collection_2)
collection.extend(collection_3)
collection.extend(collection_4)
best_set = [
    'ADAUSDT', 
    'XRPUSDT', 
    'LINKUSDT', 
    'MATICUSDT', 
    'UNIUSDT',
    'ATOMUSDT', 
    'FILUSDT', 
    'ALGOUSDT', 
    'MANAUSDT',
    'KAVAUSDT',
    'GALAUSDT', 
    'DYDXUSDT', 
    'DOGEUSDT', 
    'SOLUSDT',
    'LTCUSDT',
    'XLMUSDT',
    'AVAXUSDT',
    'GRTUSDT', 
    'SNXUSDT', 
    'STXUSDT', 
    'EOSUSDT',
    'SANDUSDT',
    'THETAUSDT',
    'AXSUSDT',
    'NEOUSDT',
    'RUNEUSDT', 
    'APEUSDT', 
    'SUIUSDT', 
    ]

names = [name for name in best_set if name not in collection]

for name in names:
    print(name)

print('----------------------------------------------')

names = [name for name in collection if name not in best_set]

for name in names:
    print(name)