```py
from keras.datasets import boston_housing, mnist, cifar10, imdb

(x_train1, y_train1), (x_test1, y_test1) = boston_housing.load_data()
(x_train2, y_train2), (x_test2, y_test2) = mnist.load_data()
(x_train3, y_train3), (x_test3, y_test3) = cifar10.load_data()
(x_train4, y_train4), (x_test4, y_test4) = imdb.load_data(num_words=20000)

num_classes = 10
```

```py
from keras.layers import Input

inputs = Input(shape=(784,))
```

```py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

x_train = np.random.random((1000, 100))
y_train = np.random.randint(2, size=(1000, 1))

x_test = np.random.random((500, 100))
y_test = np.random.randint(2, size=(500, 1))

model = Sequential()
model.add(Dense(32, input_dim=100, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Binary Classification
adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])
# Multi-Class Classfication
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Regression
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

early_stopping_monitor = EarlyStopping(patience=2)
model.fit(data, labels, epochs=10, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor])

score = model.evaluate(x_test, y_test, batch_size=32)

predictions = model.predict(data)
```

```py
from keras.layers import Embedding, LSTM

model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
```

```py
model.output_shape
model.summary()

config = model.get_config()
model = Model.from_config(config)
model = Sequential.from_config(config)

weights = model.get_weights()
model.set_weights(weights)

model.save_weights(filepath)
model.load_weights(filepath)
```

```py
from keras.models import load_model

model.save('model_file.h5')
my_model = load_model('model_file.h5')
```

```py
from keras.utils import plot_model

plot_model(model, to_file='model.png')
```
