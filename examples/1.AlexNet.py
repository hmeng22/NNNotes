from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2

# Intpu image is 224x224
image_shape = (224, 224, 3)

alexNet = Sequential()

# Layer 1 : Conv2D -> (LRN) -> MaxPooling
alexNet.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=image_shape, padding='valid', activation='relu', kernel_initializer='uniform'))
alexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Layer 2 : Conv2D -> (LRN) -> MaxPooling
alexNet.add(Conv2D(256, (5, 5), (1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
alexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Layer 3 : Conv2D
alexNet.add(Conv2D(384, (3, 3), (1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

# Layer 4 : Conv2D
alexNet.add(Conv2D(384, (3, 3), (1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

# Layer 5 : Conv2D -> MaxPooling
alexNet.add(Conv2D(256, (3, 3), (1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
alexNet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Layer 6 : Flatten -> FullyConnected -> Dropout
alexNet.add(Flatten())
alexNet.add(Dense(4096, activation='relu'))
alexNet.add(Dropout(0.5))

# Layer 7 : FullyConnected -> Dropout
alexNet.add(Dense(4096, activation='relu'))
alexNet.add(Dropout(0.5))

# Layer Output : FullyConnected -> Softmax
alexNet.add(Dense(1000, activation='softmax'))

alexNet.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
alexNet.summary()
