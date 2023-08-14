import tensorflow as tf
import numpy as np
from contextlib import redirect_stdout
import os


def make_cnn(input_shape, num_classes):
    print('building cnn model ===========')

    input_layer = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3,  input_shape=input_shape, padding="same")(input_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(gap)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)





def make_lstm(x_train, num_classes,num_units,work_dir):
    # lstm
    print('building lstm model ===========')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(num_units, input_shape=x_train.shape[1:]))
    print('x_train.shape[1:] - base_model')
    print(x_train.shape[1:])
    model.add(tf.keras.layers.Dropout(0.8))
    model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.build(x_train.shape[1:])
    print('dropout ===== 0.8')
    model.summary()

    with open(os.path.join(work_dir,'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()


    return model


# https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras