import tensorflow as tf

mnist = tf.keras.datasets.mnist

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    print("no available gpu")


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist dataset contains 60,000 training samples and 10,000 test samples
x_train, x_test = x_train / 255.0, x_test / 255.0
# scales the values obtained by the image to a range between 1 and 0

# Check the size of the training dataset
print("Training dataset size:", y_train.shape[0])

# Check the size of the testing dataset
print("Testing dataset size:", x_test.shape[0])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),  # declare the input shape
    tf.keras.layers.Flatten(),  # multiplies the arguments of the output shape
    tf.keras.layers.Dense(128, activation='relu'),  # changes the shape of the output
    tf.keras.layers.Dropout(0.2),  # multiplies the inputs by 1/ (1-rate)
    tf.keras.layers.Dense(10)
])
# for each example the model returns a vector of logits and log-odds, one for each class
predictions = model(x_train[:1]).numpy()
print(predictions)
# creates a vector of raw(non-normalized) predictions

soft = tf.nn.softmax(predictions).numpy()
# creates a vector of normalized probabilities one value for each possible class
print(soft)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss functions takes a vector of ground truth and logits
              # returns a scalar loss for each example
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32,  validation_data=(x_test, y_test))
# batch size divides the total number of training samples in this instance it is 60,000/32

model.evaluate(x_test, y_test)


