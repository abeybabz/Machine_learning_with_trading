import MetaTrader5 as mt
from datetime import datetime
import time
import schedule
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import talib
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def job():
    mt.initialize()
    login = 
    password= ''
    server= ''
    mt.login(login, password, server)
    print(mt.account_info())
    
job()


def job1():
    
    import random
    random.seed(42)
    
    eurusd_rates = mt.copy_rates_from("EURUSD", mt.TIMEFRAME_M15, datetime(2030,4,30), 91)
    Df = pd.DataFrame(eurusd_rates)
    # convert time in seconds into the datetime format
    Df['time']=pd.to_datetime(Df['time'], unit='s')
    
    Df = Df[['open', 'high', 'low', 'close']]
    
    print(Df.tail())
    Df['H-L'] = Df['high'] - Df['low']
    Df['O-C'] = Df['open'] - Df['close']
    Df['3day MA'] = Df['close'].shift(1).rolling(window=3).mean()
    Df['10day MA'] =Df['close'].shift(1).rolling(window = 10).mean()
    Df['30day MA'] = Df['close'].shift(1).rolling(window = 30).mean()
    

    Df['Std_dev'] = Df['close'].rolling(5).std()    
    Df['RSI'] = talib.RSI(Df['close'].values, timeperiod = 9)
    Df['Williams %R'] = talib.WILLR(Df['high'].values, Df['low'].values, Df['close'].values, 7)
    
    Df['Price_Rise'] = np.where(Df['close'].shift(-1) > Df['close'], 1, 0)
    
    
    Df = Df.dropna()
    print(Df.shape)
    
    Df = Df.iloc[:-1,:]
    
    X=Df.iloc[:, 4:-1]
    y=Df.iloc[:, -1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    
    #split = int(len(Df)*0.8)
    #X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    
    model= Sequential()
    model.add(Dense(units=128, kernel_initializer= 'uniform', activation='relu', input_dim= X.shape[1]))
    model.add(Dense(units=128, kernel_initializer= 'uniform', activation='relu'))
    model.add(Dense(units=8, kernel_initializer= 'uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer= 'uniform', activation='sigmoid'))
    
    
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    
    earlystopping_callback = EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=3, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
   )
    
    
    model.fit(X_train, y_train, batch_size=5, epochs= 100, validation_split=0.1, callbacks=[earlystopping_callback])
    test_loss, test_acc= model.evaluate(X_test, y_test, verbose=1)
    print('Test accuracy:', test_acc)
    
    y_pred= model.predict(X_test)
    y_pred = (y_pred > 0.5)
    Df['y_pred'] = np.NaN
    Df.iloc[(len(Df) - len(y_pred)):,-1:] = y_pred
    trade_Df = Df.dropna()
    
    # Computing strategy returns
    trade_Df['Next_Returns'] = 0
    trade_Df['Next_Returns'] = np.log(trade_Df['close']/trade_Df['close'].shift(1))
    trade_Df['Next_Returns'] = trade_Df['Next_Returns'].shift(-1)
    
    trade_Df['Strategy_Returns'] = 0
    trade_Df['Strategy_Returns'] = np.where(trade_Df['y_pred'] == True, trade_Df['Next_Returns'], - trade_Df['Next_Returns'])
    
    trade_Df['Cumulative Market Returns'] = np.cumsum(trade_Df['Next_Returns'])
    trade_Df['Cumulative Strategy Returns'] = np.cumsum(trade_Df['Strategy_Returns'])
    
    plt.figure()
    plt.plot(trade_Df['Cumulative Market Returns'], color='r', label='Market Returns')
    plt.plot(trade_Df['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
    plt.legend()
    plt.show()
   
    
    print("Value:", trade_Df['Price_Rise'].iloc[-1])
    print("Pred:", trade_Df['y_pred'].iloc[-1])
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    trade_Df['pred']= le.fit_transform(trade_Df['y_pred'])
    print('pred_1',trade_Df['pred'].iloc[-1])
  
    
    positions = mt.positions_total()
    balance = mt.account_info().balance
    Price_Rise= trade_Df['Price_Rise'].iloc[-1]
    pred_1 = trade_Df['pred'].iloc[-1]
    
    MA3 = trade_Df['3day MA'].iloc[-1]
    MA10 = trade_Df['10day MA'].iloc[-1]
    MA30 = trade_Df['30day MA'].iloc[-1]
    
    
    lot= round((balance*0.04)/100,2)
    symbol = "EURUSD"
    
    if  (test_acc > 0.63) and (Price_Rise == 0) and (pred_1 == 0) and (positions == 0) and (MA3 < MA10) and (MA10 < MA30):
        point = mt.symbol_info(symbol).point
        price = mt.symbol_info_tick(symbol).ask        
        deviation = 20
        request1 = {
            "action": mt.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt.ORDER_TYPE_SELL,
            "price": price,
            "deviation": deviation,
            "sl": price + 100 * point,
            "tp": price - 300 * point,
            "magic": 2,
            "comment": "python script open(N_Network)",
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_FOK
        }
        # send a trading request
        mt.order_send(request1)
   
    if (test_acc > 0.63) and (Price_Rise == 1) and (pred_1 == 1) and (positions == 0) and (MA3 > MA10) and (MA10 > MA30):
        point = mt.symbol_info(symbol).point
        price = mt.symbol_info_tick(symbol).bid        
        deviation = 20
        request1 = {
            "action": mt.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt.ORDER_TYPE_BUY,
            "price": price,
            "deviation": deviation,
            "sl": price - 100 * point,
            "tp":price + 300 * point,
            "magic": 2,
            "comment": "python script open(N_Network)",
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_FOK
        }
        # send a trading request
        mt.order_send(request1)
            

schedule.every().hour.at(":00").do(job1)
#schedule.every().hour.at(":05").do(job1)
#schedule.every().hour.at(":10").do(job1)
schedule.every().hour.at(":15").do(job1)
#schedule.every().hour.at(":20").do(job1)
#schedule.every().hour.at(":25").do(job1)
schedule.every().hour.at(":30").do(job1)
#schedule.every().hour.at(":35").do(job1)
#schedule.every().hour.at(":40").do(job1)
schedule.every().hour.at(":45").do(job1)
#schedule.every().hour.at(":50").do(job1)
#schedule.every().hour.at(":55").do(job1)







#schedule.every(5).minutes.do(job1)

#schedule.every().day.at("21:30").do(shutdown)
while True:
    schedule.run_pending()
    time.sleep(5)




