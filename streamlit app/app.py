
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dr
import keras
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
st.subheader("keep the dates before 30 july 2022")
start_date = st.text_input('enter start date in format YYYY-MM-DD ', '2012-06-30')
#end_date = '2022-06-30'
end_date =  st.text_input('enter end date in format YYYY-MM-DD ', '2022-06-30')
steps =50

st.title('stock trend prediction')

user_input = st.text_input('enter stock ticker' , 'SBIN.NS')

data = dr.DataReader(user_input,'yahoo', start_date, end_date)

st.subheader('description of data')
st.write(data.describe())

st.subheader('closing price vs time chart with moving average 100 days')
ma100 = data.Close.rolling(100).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(data.Close)
plt.plot(ma100)
st.pyplot(fig)



data_train = pd.DataFrame(data.Close[:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])


scaler = MinMaxScaler(feature_range =(0,1))
scaled_train = scaler.fit_transform(data_train)
scaled_test = scaler.fit_transform(data_test)


def dataset_lstm(dataset,steps):
  x_train,y_train =[],[]
  for i in range(len(dataset)-steps-1):
    a = dataset[i:(i+steps),0]
    x_train.append(a)
    y_train.append(dataset[i+steps,0])
  return np.array(x_train),np.array(y_train)

x_train,y_train = dataset_lstm(scaled_train,steps)
x_test,y_test = dataset_lstm(scaled_test,steps)


model = load_model('time_series_model (1).h5')


previous_50_data = data_train.tail(50)
final_test_data = previous_50_data.append(data_test,ignore_index= True)

final_test_data = scaler.fit_transform(final_test_data)

final_x_test, final_y_test = dataset_lstm(final_test_data,steps)

y_predict = model.predict(final_x_test)

final_y_test = scaler.inverse_transform(final_y_test.reshape(-1,1))
y_predict = scaler.inverse_transform(y_predict)



fig = plt.figure(figsize=(12,6))
st.subheader('plot of testing data ')
plt.plot(final_y_test, 'r',label ='actual values')
plt.plot(y_predict, 'b',label = 'predicted values')
plt.xlabel("Time")
plt.ylabel('closing price')
plt.legend()
st.pyplot(fig)


train_predict = model.predict(x_train)
train_predict =scaler.inverse_transform(train_predict)




st.subheader('final plot of actual values and predicted values(along with train and test)')

fig2 = plt.figure(figsize=(12,6))

plt.plot(np.arange(0,len(x_train)),train_predict,'r',label = 'training data prediction')
plt.plot(np.arange(len(x_train),(len(x_train)+len(x_test)+50)), y_predict, 'b',label = 'test prediction value')

plt.plot(data.Close.values, 'g',label ='actual values')
plt.xlabel('time')
plt.ylabel('closing price')
plt.legend()
st.pyplot(fig2)











