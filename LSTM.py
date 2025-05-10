from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np




def lstm(stock_df):
    data = stock_df.filter(["Adj Close"])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .80 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    # -> (B, 60, 1)
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1), use_bias = False))
    # -> (B, 60, 128)
    model.add(LSTM(64, return_sequences=False, use_bias = False))
    # -> (B, 64)
    model.add(Dense(25))
    # -> (B, 25)
    model.add(Dense(1))
    # -> (B, 1)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=25)

    # Create the testing data set
    test_data = scaled_data[training_data_len:, :] 
    #To prevent information leakage, we split the dataset into training (80%) 
    # and testing (20%) sets. When constructing the test sequences (x_test), 
    # only past values from the test period are used, starting from training_data_len - 60.
    # This ensures all predictions are made using only past information.
    # Create the data sets x_test and y_test
    x_test = []
    y_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
        y_test.append(test_data[i, 0])
        
    # Convert the data to a numpy array
    x_test, y_test = np.array(x_test), np.array(y_test)
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    preds = model.predict(x_test)
    preds = scaler.inverse_transform(preds)
    
    return preds

