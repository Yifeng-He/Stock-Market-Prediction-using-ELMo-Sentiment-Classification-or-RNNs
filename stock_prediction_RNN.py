'''
This project aims to predict the trend for a stock in the next day based on the 
historical prices.
'''

import numpy as np
import tensorflow as tf
from pandas_datareader import data as pdr


# function to get the historical data, default length = 1 year
def get_stock_historical_data(stock_symbol, look_back_length=2, end_date_str='today', end_date_shift=0): 
    end_date = np.datetime64(end_date_str)-end_date_shift
    start_date = end_date - 365*look_back_length
    return pdr.get_data_yahoo(stock_symbol, start_date, end_date)


def prepare_data(df_data, sliding_window=2, test_ratio=0.05, scaler=0.01):
    df_data['Open1'] = df_data['Open']*scaler
    df_data['Close1'] = df_data['Close']*scaler
    df_data['percent'] = df_data.apply(lambda row: (row.Open-row.Close)/row.Open, axis=1)
    df_data['percent1'] = df_data.apply(lambda row: (row.Open1-row.Close1)/row.Open1, axis=1)
    len_data = len(df_data)
    train_lenghth = int(len_data*(1-test_ratio))
    range_index = range(sliding_window, len_data)
    y = np.array( [df_data['percent1'].iloc[k] for k in range_index]).reshape((-1,1))
    x2 = np.array([df_data['Open1'].iloc[k] for k in range_index]).reshape((-1,1))
    x1 = np.array([ [ [df_data['Open1'].iloc[i], df_data['Close1'].iloc[i]] for i in range(k-sliding_window,k)] for k in range_index]) 
    y_train = y[:train_lenghth]
    y_test = y[train_lenghth:]
    x1_train = x1[:train_lenghth]
    x1_test = x1[train_lenghth:]
    x2_train = x2[:train_lenghth]
    x2_test = x2[train_lenghth:]
    return (x1_train, x1_test, x2_train, x2_test, y_train, y_test)



if __name__=='__main__':       
    # random seed
    np.random.seed(1234)
    # load the data
    stock_symbol = 'DGAZ'
    list_profits = []
    for end_date_shift in range(10):
        df_data = get_stock_historical_data(stock_symbol, end_date_shift=end_date_shift) 
        sliding_window = 5
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = prepare_data(df_data=df_data, sliding_window=sliding_window)
        
        input1 = tf.keras.Input(shape=(sliding_window,2))
        input2 = tf.keras.Input(shape=(1,))
        lstm2 = tf.keras.layers.LSTM(16)(input1)
        com1 = tf.keras.layers.concatenate([lstm2, input2])
        output = tf.keras.layers.Dense(1)(com1)
        model = tf.keras.Model(inputs=[input1, input2], outputs=[output])
        
        model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer(0.001))
        model.fit([x1_train, x2_train], [y_train], epochs=500, batch_size=1024)
        
        # prediction
        predictions = model.predict([x1_test, x2_test])
        actions = [1 if s>0 else -1 for s in predictions]
        profits = [actions[k]*y_test[k]*100 for k in range(len(predictions))]
        list_profits.append( [end_date_shift, actions[0], profits[0]])
