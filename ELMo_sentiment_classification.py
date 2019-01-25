"""
This project aims to predict the trend for Dow Jones Industrial Average (DJIA)
in the next day based on the sentiment classification on the daily news. 
"""

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K


# ELMo pre-trained model
url = 'https://tfhub.dev/google/elmo/2'
embed = hub.Module(url)

# the embedded layer
def ElmoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature='default', as_dict=True)['default']
        
# prepare the data
data_url='https://raw.githubusercontent.com/Yifeng-He/Stock-Market-Prediction-using-ELMo-Sentiment-Classification-or-RNNs/master/Combined_News_DJIA.csv'
df_data = pd.read_csv(data_url, encoding = "ISO-8859-1")
df_data.Date = pd.to_datetime(df_data.Date)
df_data['text']=df_data.apply(lambda row: row['Top1']+' '+row['Top2']+' '+row['Top3']
  +' '+row['Top4']+' '+row['Top5']+' '+row['Top6']+' '+row['Top7']+' '+row['Top8']+
  ' '+row['Top9']+' '+row['Top10'], axis=1)
df_y = pd.get_dummies(df_data.Label)
train_len = 1500
x_train = df_data['text'].values[:train_len]
x_test = df_data['text'].values[train_len:]
y_train = df_y.values[:train_len]
y_test = df_y.values[train_len:]

# sentiment classification model
input_text = tf.keras.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
dense=tf.keras.layers.Dense(256, activation='relu')(embedding)
pred=tf.keras.layers.Dense(2, activation='softmax')(dense)
model = tf.keras.Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(0.001), metrics=['accuracy'])

with tf.Session() as sess:
  K.set_session(sess)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  history = model.fit(x_train, y_train, epochs=1, batch_size=32)
  accuracy = model.evaluate(x_test, y_test)
  print(accuracy)
