import tensorflow as tf
from data_preprocessing import load_data, augment_data
from model import create_model

(x_train, y_train), (x_test, y_test) = load_data()
datagen = augment_data()
model = create_model()

# Train model
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test),
                    epochs=20)

# Save model
model.save("models/cifar10_model.h5")
