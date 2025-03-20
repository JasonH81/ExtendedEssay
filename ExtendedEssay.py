import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

# Data Set-up
img_rows, img_cols = 28, 28
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Model set-up
epochs = 1
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(data_augmentation)
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=64, epochs=epochs, validation_data=(test_images, test_labels), shuffle=True)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

image = train_images[10042]
label = np.argmax(train_labels[10042])
image_batch = tf.expand_dims(image, axis=0)
result = data_augmentation(image_batch)
result = result.numpy()
_ = plt.imshow(result.reshape(28, 28), cmap='gray')
#_ = plt.imshow(image)
plt.title(f"Original (Label: {label})")
plt.axis("off")
plt.show()