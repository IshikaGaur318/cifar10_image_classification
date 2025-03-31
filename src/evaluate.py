import tensorflow as tf
from data_preprocessing import load_data

(x_train, y_train), (x_test, y_test) = load_data()
model = tf.keras.models.load_model("models/cifar10_model.h5")

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
