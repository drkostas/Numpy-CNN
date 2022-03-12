import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from src import generateExample1, generateExample2,generateExample3

def tensorExample1():
    # Create a feed forward network
    model = Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(1, 3, input_shape=(5, 5, 1), activation='sigmoid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    # Call weight/data generating function
    l1, l1b, l2, l2b, input, targets = generateExample1()

    # Set weights to desired values

    # setting weights and bias of first layer.
    w1 = l1.reshape(3, 3, 1, 1)
    model.layers[0].set_weights(
        [w1, np.array([l1b[0]])]) # Shape of weight matrix is (w,h,input_channels,kernels)

    # setting weights and bias of fully connected layer.
    model.layers[2].set_weights([np.transpose(l2), l2b])

    # Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    img = np.expand_dims(input, axis=(0, 3))

    # print needed values.
    print("----------- Tensor Model -----------")
    np.set_printoptions(precision=5)
    print('model output before:')
    print(model.predict(img))
    sgd = optimizers.SGD(learning_rate=100)
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(img, targets, batch_size=1, epochs=1)
    print('model output after:')
    print(model.predict(img))

    print('1st convolutional layer, kernel weights:')
    print(np.squeeze(model.get_weights()[0][:, :, 0, 0]))
    print('1st convolutional layer, kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))

    print('fully connected layer weights:')
    print(np.squeeze(model.get_weights()[2]))
    print('fully connected layer bias:')
    print(np.squeeze(model.get_weights()[3]))

def tensorExample2():
    # Create a feed forward network
    model = Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(2, 3, input_shape=(7, 7, 1), activation='sigmoid'))
    model.add(layers.Conv2D(1, 3, activation='sigmoid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    # Call weight/data generating function
    l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input, output = generateExample2()

    # Set weights to desired values

    # setting weights and bias of first layer.
    l1k1 = l1k1.reshape(3, 3, 1, 1)
    l1k2 = l1k2.reshape(3, 3, 1, 1)

    w1 = np.concatenate((l1k1, l1k2), axis=3)
    model.layers[0].set_weights(
        [w1, np.array([l1b1[0], l1b2[0]])])  # Shape of weight matrix is (w,h,input_channels,kernels)

    # setting weights and bias of second layer.
    l2c1 = l2c1.reshape(3, 3, 1, 1)
    l2c2 = l2c2.reshape(3, 3, 1, 1)

    w1 = np.concatenate((l2c1, l2c2), axis=2)
    model.layers[1].set_weights([w1, l2b])

    # setting weights and bias of fully connected layer.
    model.layers[3].set_weights([np.transpose(l3), l3b])

    # Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    img = np.expand_dims(input, axis=(0, 3))

    # print needed values.
    np.set_printoptions(precision=5)
    print('model output before:')
    print(model.predict(img))
    sgd = optimizers.SGD(learning_rate=100)
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(img, output, batch_size=1, epochs=1)
    print('model output after:')
    print(model.predict(img))

    print('1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:, :, 0, 0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))

    print('1st convolutional layer, 2nd kernel weights:')
    print(np.squeeze(model.get_weights()[0][:, :, 0, 1]))
    print('1st convolutional layer, 2nd kernel bias:')
    print(np.squeeze(model.get_weights()[1][1]))

    print('2nd convolutional layer weights:')
    print(np.squeeze(model.get_weights()[2][:, :, 0, 0]))
    print(np.squeeze(model.get_weights()[2][:, :, 1, 0]))
    print('2nd convolutional layer bias:')
    print(np.squeeze(model.get_weights()[3]))

    print('fully connected layer weights:')
    print(np.squeeze(model.get_weights()[4]))
    print('fully connected layer bias:')
    print(np.squeeze(model.get_weights()[5]))


def tensorExample3():
    # Create a feed forward network
    model = Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(2, 3, input_shape=(8, 8, 1), activation='sigmoid'))
    model.add(layers.MaxPooling2D(pool_size = (2,2), strides=(1,1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    # Call weight/data generating function
    l1k1, l1k2, l1b1, l1b2, l3, l3b, input, output = generateExample3()

    # Set weights to desired values

    # setting weights and bias of first layer.
    l1k1 = l1k1.reshape(3, 3, 1, 1)
    l1k2 = l1k2.reshape(3, 3, 1, 1)

    w1 = np.concatenate((l1k1, l1k2), axis=3)
    model.layers[0].set_weights(
        [w1, np.array([l1b1[0], l1b2[0]])])  # Shape of weight matrix is (w,h,input_channels,kernels)


    # setting weights and bias of fully connected layer.
    model.layers[3].set_weights([np.transpose(l3), l3b])

    # Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    img = np.expand_dims(input, axis=(0, 3))

    # print needed values.
    np.set_printoptions(precision=5)
    print('model output before:')
    print(model.predict(img))
    sgd = optimizers.SGD(learning_rate=100)
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(img, output, batch_size=1, epochs=1)
    print('model output after:')
    print(model.predict(img))

    print('1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:, :, 0, 0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))

    print('1st convolutional layer, 2nd kernel weights:')
    print(np.squeeze(model.get_weights()[0][:, :, 0, 1]))
    print('1st convolutional layer, 2nd kernel bias:')
    print(np.squeeze(model.get_weights()[1][1]))

    print('fully connected layer weights:')
    print(np.squeeze(model.get_weights()[2]))
    print('fully connected layer bias:')
    print(np.squeeze(model.get_weights()[3]))