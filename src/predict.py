import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data

(x_train, y_train), (x_test, y_test) = load_data()
model = tf.keras.models.load_model("models/cifar10_model.h5")

# Predict on a single image
index = np.random.randint(0, len(x_test))
img = x_test[index]
pred = model.predict(img.reshape(1,32,32,3))

plt.imshow(img)
plt.title(f"Predicted: {np.argmax(pred)}")
plt.show()
