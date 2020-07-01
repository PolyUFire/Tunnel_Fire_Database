"""
predict the characteristics of tunnel fire

@author: wu
"""

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

import crt_folder
import xlsxwriter
import xlrd



"""
0. Initialize
set the dictionary
"""

# fix random seed
np.random.seed(1234)

father_path = ('C:\\Users\\wu\\Google Drive\\Desktop')
path_data =(father_path + '\\devc')
output_dir = father_path + '\\output'
crt_folder.crt_folder(output_dir)
log_dir = father_path + '\\log'
crt_folder.crt_folder(log_dir)

"""
1. Read the database
"""
read_data = xlrd.open_workbook(father_path + '\\database.xlsx')

sheet_cases = read_data.sheets()[0]
nrows = sheet_cases.nrows
All_data = []
for i in np.arange(nrows-1):
    All_data.append(sheet_cases.row_values(i+1))

"""
2. Form the database
scale, shuffle and split
"""

# scale the labels
MMS = MinMaxScaler()
All_data = MMS.fit_transform(np.array(All_data, dtype = np.float))
Train_df = All_data[:,0:-1]
Truth_df = All_data[:,-1].reshape(-1,1)

# shuffle the data
index = [i for i in range(len(Truth_df))]
np.random.shuffle(index)
Truth_df = Truth_df[index]
Train_df = Train_df[index]
# split the database
test_ratio = 0.2
Truth_df_test = Truth_df[:int(test_ratio*Truth_df.shape[0])]
Train_df_test = Train_df[:int(test_ratio*Train_df.shape[0])]

Truth_df = Truth_df[int(test_ratio*Truth_df.shape[0]):]
Train_df = Train_df[int(test_ratio*Train_df.shape[0]):]

"""
3. Train the model
(1) define the criteria of accuracy
(2) build up the model
(3) fit the models
"""
def r2_total(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model = Sequential()
model.add(Dense(6, input_dim = 6, activation = 'relu'))
model.add(Dense(6, input_dim = 6, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
model.compile(loss = 'mse', optimizer = 'adam', metrics=[r2_total])
model.summary()

history = model.fit(Train_df, Truth_df, epochs=100, validation_split=0.25, verbose=1)

"""
4. Evaluate the model
"""
results_reg = np.array(MMS.inverse_transform(np.concatenate((Train_df,model.predict(Train_df)),axis=1)))[:,-1]
Truth_df = np.array(MMS.inverse_transform(np.concatenate((Train_df,Truth_df),axis=1)))[:,-1]

loss_test, r2_test = model.evaluate(Train_df_test,Truth_df_test)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
plt.plot(history.history['r2_total'])
plt.plot(history.history['val_r2_total'])
plt.show()

plt.plot(Train_df[:,-1],results_reg, '*')

plt.plot(Train_df[:,-1],Truth_df, 'o')
plt.show()

plt.plot(Truth_df,results_reg, 'o')
plt.show()
