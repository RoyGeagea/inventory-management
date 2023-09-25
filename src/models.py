from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def get_mlp_model(input_shape,hidden_layer_one,dropout_one,hidden_layer_two,dropout_two):
    model = Sequential()
    model.add(Dense(hidden_layer_one, activation='elu', input_shape=input_shape))
    if dropout_one !=0:
        model.add(Dropout(dropout_one))
    model.add(Dense(hidden_layer_two, activation='elu'))
    if dropout_two !=0:
        model.add(Dropout(dropout_two))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer="Adam")

    return model