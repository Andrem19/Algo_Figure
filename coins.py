


oldCoins = ['ETHUSDT', 'DOTUSDT', 'BNBUSDT', 'ADAUSDT', 'BTCUSDT', 'XRPUSDT', 'LINKUSDT', 'MATICUSDT', 'UNIUSDT',
     'ATOMUSDT', 'FILUSDT', 'VETUSDT', 'ALGOUSDT', 'FTMUSDT', 'MANAUSDT', 'KAVAUSDT', 'GALAUSDT', 'DYDXUSDT']

nowSet = [
    'FILUSDT', 
    'XRPUSDT', 
    'GALAUSDT', 
    'KAVAUSDT', 
    'ALGOUSDT', 
    'FTMUSDT', 
    'ADAUSDT', 
    'ATOMUSDT', 
    'SOLUSDT', 
    'GALAUSDT', 
    'INJUSDT', 
    'GRTUSDT', 
    'DOGEUSDT',
    'AXSUSDT',
    ]
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
collection = ['TRXUSDT', 'XMRUSDT', 'RNDRUSDT', 'MINAUSDT', 'SUIUSDT', 'XRPUSDT']
new_collection = [
    'MATICUSDT', 
    'UNIUSDT', 
    'LTCUSDT', 
    'FILUSDT', 
    'AXSUSDT', 
    'SOLUSDT', 
    'GALAUSDT', 
    'INJUSDT', 
    'GRTUSDT', 
    'DOGEUSDT', 
    'SNXUSDT', 
    'APTUSDT', 
    'NEOUSDT']

coins_to_add = [
    'XLMUSDT', 
    'AVAXUSDT', 
    'STXUSDT', 
    'SANDUSDT', 
    'THETAUSDT', 
    'RNDRUSDT', 
    'APEUSDT', 
    'SUIUSDT', 
    'XRPUSDT', 
    'ATOMUSDT', 
    'ALGOUSDT', 
    'DYDXUSDT', 
    'MANAUSDT', 
    'KAVAUSDT'
    ]

newCoins = [    
    'DOGEUSDT',
    'SOLUSDT',
    'TRXUSDT',
    'LTCUSDT',
    'XLMUSDT',
    'AVAXUSDT',
    'XMRUSDT',
    'HBARUSDT',
    'QNTUSDT',
    'APTUSDT',
    'ARBUSDT',
    'AAVEUSDT',
    'GRTUSDT',
    'SNXUSDT',
    'STXUSDT',
    'EOSUSDT',
    'EGLDUSDT',
    'SANDUSDT',
    'THETAUSDT',
    'INJUSDT',
    'RNDRUSDT',
    'AXSUSDT',
    'NEOUSDT',
    'RUNEUSDT',
    'FLOWUSDT',
    'APEUSDT',
    'CHZUSDT',
    'KLAYUSDT',
    'FXSUSDT',
    'MINAUSDT',
    'CRVUSDT',
    'SUIUSDT',
    'DASHUSDT']

c_coins = [
    'MATICUSDT', 
    'UNIUSDT', 
    'FILUSDT', 
    'SOLUSDT', 
    'GALAUSDT', 
    'GRTUSDT', 
    'DOGEUSDT',
    'AVAXUSDT', 
    'SANDUSDT', 
    'THETAUSDT',  
    'XRPUSDT', 
    'ATOMUSDT', 
    'ALGOUSDT', 
    'MANAUSDT', 
]
test = [
    'MATICUSDT', 
    'UNIUSDT', 
    'FILUSDT', 
]
fin = [
    # 'SANDUSDT',
    # 'THETAUSDT',
    # 'INJUSDT',
    # 'RNDRUSDT',
    # 'AXSUSDT',
    # 'NEOUSDT',
    # 'RUNEUSDT',
    # 'FLOWUSDT',
    # 'APEUSDT',
    # 'CHZUSDT',
    # 'KLAYUSDT',
    # 'FXSUSDT',
    # 'MINAUSDT',
    # 'CRVUSDT',
    # 'SUIUSDT',
    # 'DASHUSDT'
]
all_coins = oldCoins + newCoins
prep_coins = [x for x in oldCoins if x not in new_collection]
def all_coins_without() -> list:
    coin_list = oldCoins + newCoins
    coin_list.remove('FXSUSDT')
    coin_list.remove('FLOWUSDT')
    coin_list.remove('XTZUSDT')
    print(coin_list)
    return coin_list