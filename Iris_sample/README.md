Compare SGD and Adam by using the sample module of the neural network.

This sample running on the kaggle platform. Download the data source from https://www.kaggle.com/uciml/iris.

# Read data
data = pd.read_csv("/kaggle/input/iris/Iris.csv")

# Create X_data
X_data = (data.drop(columns=['Id', 'Species'])).values

# Create Y_data
y = data.Species
classes_dict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
y = y.map(classes_dict).values
Y_data = to_categorical(y, num_classes=3)


# Split training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

#scaling data
scaler = MinMaxScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)


# Create the module
model = keras.Sequential(
        [
            layers.Dense(10, input_shape=(4,), name="layer1"),
            layers.Dense(10, activation="relu", name="layer2"),
            layers.Dense(8, activation="relu", name="layer3"),
            layers.Dense(3, activation="softmax", name="layer4"),
        ], name=name
    )
 
 
# Get loss and accuracy
model.compile(loss = 'categorical_crossentropy' , optimizer = optimizers[i] , metrics = ['accuracy'] )
history = model.fit(x_train_scaled, Y_train, epochs = 120, batch_size=1, verbose=0, validation_data=(x_test_scaled, Y_test))
loss, accuracy = model.evaluate(x_test_scaled, Y_test)
print(loss, accuracy)



