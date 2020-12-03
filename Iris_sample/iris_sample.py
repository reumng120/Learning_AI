import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

keras.backend.clear_session()
data = pd.read_csv("/kaggle/input/iris/Iris.csv")


# Convert List to pandas Series sample x = pd.Series(X_data)
# Create X_data. Drop Id and Species and get all values
X_data = (data.drop(columns=['Id', 'Species'])).values

# Create Y_data
y = data.Species
classes_dict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
y = y.map(classes_dict).values
Y_data = to_categorical(y, num_classes=3)

# print(X_data)
# print(Y_data)
# print(X_data.shape)
# print(Y_data.shape)

# Split training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

#scaling data
scaler = MinMaxScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)


def my_neutral(name):
    model = keras.Sequential(
        [
            layers.Dense(10, input_shape=(4,), name="layer1"),
            layers.Dense(10, activation="relu", name="layer2"),
            layers.Dense(8, activation="relu", name="layer3"),
            layers.Dense(3, activation="softmax", name="layer4"),
        ], name=name
    )
    return model
    

def create_modules(nodes, input_shape, output_dim, n, module_name):
    model = keras.Sequential(name=module_name)
    for i in range(n):
        if i == 0:
            model.add(layers.Dense(nodes, input_shape=input_shape, name="layer{}".format(i)))
        else:
             model.add(layers.Dense(nodes, activation="relu", name="layer{}".format(i)))
        
    model.add(layers.Dense(output_dim, activation="softmax", name="layer{}".format(n)))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    return model



optimizers = ['adam', 'sgd']
n = len(optimizers)
modules = []
history_dict = {}
for i in range(n):
    modules.append(my_neutral(optimizers[i]))



for i, model in enumerate(modules):
    print("Compiling", optimizers[i])
    model.compile(loss = 'categorical_crossentropy' , optimizer = optimizers[i] , metrics = ['accuracy'] )
    history = model.fit(x_train_scaled, Y_train, epochs = 120, batch_size=1, verbose=0, validation_data=(x_test_scaled, Y_test))
    loss, accuracy = model.evaluate(x_test_scaled, Y_test)
    print(loss, accuracy)
    history_dict[model.name] = [history, model]
    
    # Draw Figure
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    for model_name in history_dict:
        val_accurady = history_dict[model_name][0].history['val_accuracy']
        val_loss = history_dict[model_name][0].history['val_loss']
        ax1.plot(val_accurady, label=model_name)
        ax2.plot(val_loss, label=model_name)

    ax1.set_ylabel('validation accuracy')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend();

