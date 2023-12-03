import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization, LayerNormalization, Conv2D, MaxPooling2D, \
    Flatten, Dense
from tensorflow.keras.activations import relu, selu, swish
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
from tensorflow.keras.applications import VGG16, MobileNetV2
import pandas as pd


def preprocess_image(image, label):  # Функція для обробки та аугментації зображень
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / 255.0  # нормалізація значень пікселів до діапазону [0, 1]
    return image, label


def print_graphic(history):
    plt.plot(history.history['loss'], label='Навчальна вибірка')
    plt.plot(history.history['val_loss'], label='Тестова вибірка')
    plt.xlabel('epoch')
    plt.ylabel('помилка')
    plt.legend()
    plt.title('Графік помилок')
    plt.show()


def task_3(ds_train, ds_test):
    # Архітектура мережі
    experiment_1_model = models.Sequential()
    experiment_1_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    experiment_1_model.add(layers.MaxPooling2D((2, 2)))
    experiment_1_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_1_model.add(layers.MaxPooling2D((2, 2)))
    experiment_1_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_1_model.add(layers.Flatten())
    experiment_1_model.add(layers.Dense(64, activation='relu'))
    experiment_1_model.add(layers.Dense(62, activation='softmax'))

    # Компіляція моделі
    experiment_1_model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

    # Навчання моделі
    experiment_1_history = experiment_1_model.fit(ds_train, epochs=8, validation_data=ds_test)

    return experiment_1_model, experiment_1_history


def build_model(dropout, activation, use_residual, normalization, use_atrous_conv, dilation_rate=1):
    model = Sequential()

    print(f"\n\tПобудова моделі з такими параметрами:\n"
          f"Дропаут = {dropout}, функції активації = {activation}, "
          f"Residual block = {use_residual}, Normalization = {normalization}, ")
    if use_atrous_conv:
        print(f"Atrous Convolution or Dilated Convolution = {use_atrous_conv} with dilation_rate = {dilation_rate}.\n")
    else:
        print("Atrous Convolution or Dilated Convolution = False.\n")

    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 1),
                     dilation_rate=(dilation_rate, dilation_rate)))

    if normalization == 'batch':
        model.add(BatchNormalization())
    elif normalization == 'layer':
        model.add(LayerNormalization())
    elif normalization == 'group':
        model.add(tfa.layers.GroupNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation=activation, dilation_rate=(dilation_rate, dilation_rate)))

    if normalization == 'batch':
        model.add(BatchNormalization())
    elif normalization == 'layer':
        model.add(LayerNormalization())
    elif normalization == 'group':
        model.add(tfa.layers.GroupNormalization())

    model.add(MaxPooling2D((2, 2)))

    if use_residual:
        residual_block = Sequential()
        residual_block.add(
            Conv2D(64, (3, 3), activation=activation, padding='same', dilation_rate=(dilation_rate, dilation_rate)))
        if normalization == 'batch':
            residual_block.add(BatchNormalization())
        elif normalization == 'layer':
            residual_block.add(LayerNormalization())
        elif normalization == 'group':
            residual_block.add(tfa.layers.GroupNormalization())
        model.add(residual_block)

    model.add(Flatten())
    model.add(Dense(64, activation=activation))

    if dropout:
        model.add(Dropout(0.5))

    model.add(Dense(62, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, ds_train, ds_test):
    history = model.fit(ds_train, epochs=8, validation_data=ds_test)
    return history


def vgg(ds_train, ds_test):
    base_model_1 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Замороження ваг базової моделі
    for layer in base_model_1.layers:
        layer.trainable = False

    # Додавання своїх верхніх шарів
    experiment_5_model = models.Sequential()
    experiment_5_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    experiment_5_model.add(layers.MaxPooling2D((2, 2)))
    experiment_5_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_5_model.add(layers.MaxPooling2D((2, 2)))
    experiment_5_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_5_model.add(layers.Flatten())
    experiment_5_model.add(layers.Dense(64, activation='relu'))

    # Додавання верхнього шару відповідно до кількості класів у датасеті
    experiment_5_model.add(layers.Dense(62, activation='softmax'))

    # Компіляція моделі
    experiment_5_model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

    # Навчання моделі
    experiment_5_history = experiment_5_model.fit(ds_train, epochs=8, validation_data=ds_test)
    return experiment_5_model, experiment_5_history


def mobile_net(ds_train, ds_test):
    base_model_2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Замороження ваг базової моделі
    for layer in base_model_2.layers:
        layer.trainable = False

    # Додавання своїх верхніх шарів
    experiment_6_model = models.Sequential()
    experiment_6_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    experiment_6_model.add(layers.MaxPooling2D((2, 2)))
    experiment_6_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_6_model.add(layers.MaxPooling2D((2, 2)))
    experiment_6_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    experiment_6_model.add(layers.Flatten())
    experiment_6_model.add(layers.Dense(64, activation='relu'))

    # Додавання верхнього шару відповідно до кількості класів у датасеті
    experiment_6_model.add(layers.Dense(62, activation='softmax'))

    # Компіляція моделі
    experiment_6_model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

    # Навчання моделі
    experiment_6_history = experiment_6_model.fit(ds_train, epochs=8, validation_data=ds_test)
    return experiment_6_model, experiment_6_history


def create_results_table(history_list):
    table_data = {'Experiment': [], 'Train Loss': [], 'Train Accuracy': [], 'Validation Loss': [],
                  'Validation Accuracy': []}

    for i, history in enumerate(history_list):
        table_data['Experiment'].append(f'Experiment {i + 1}')
        table_data['Train Loss'].append(round(history.history['loss'][-1], 4))
        table_data['Train Accuracy'].append(round(history.history['accuracy'][-1], 4))
        table_data['Validation Loss'].append(round(history.history['val_loss'][-1], 4))
        table_data['Validation Accuracy'].append(round(history.history['val_accuracy'][-1], 4))

    results_table = pd.DataFrame(table_data)
    results_table.to_excel('results.xlsx', index=False)


def plot_all_experiments(history_list):
    plt.figure(figsize=(15, 7))
    for i, history in enumerate(history_list):
        plt.plot(history.history['loss'], label=f'Експеримент {i + 1} - Навчальна')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.figure(figsize=(15, 7))
    for i, history in enumerate(history_list):
        plt.plot(history.history['val_loss'], label=f'Експеримент {i + 1} - Тестова')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main():
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    # Обробка та Аугментація до навчальної та тестової вибірки
    ds_train = ds_train.map(preprocess_image)
    ds_test = ds_test.map(preprocess_image)
    ds_train = ds_train.shuffle(buffer_size=10000).batch(64)
    ds_test = ds_test.batch(64)

    for image, label in ds_train.take(1):
        plt.figure(figsize=(8, 8))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[i].numpy(), cmap='gray')
            plt.title(f'Label: {label[i]}')
            plt.axis('off')
        plt.show()

    history_list = []

    print('\tАрхітектура власної мережі та її навчання.')
    experiment_1_model, experiment_1_history = task_3(ds_train, ds_test)
    print_graphic(experiment_1_history)
    history_list.append(experiment_1_history)

    experiment_2_model = build_model(dropout=True, activation='relu', use_residual=True, normalization='batch',
                                     use_atrous_conv=True, dilation_rate=2)
    experiment_2_history = train_model(experiment_2_model, ds_train, ds_test)
    print_graphic(experiment_2_history)
    history_list.append(experiment_2_history)

    experiment_3_model = build_model(dropout=False, activation='selu', use_residual=False, normalization='layer',
                                     use_atrous_conv=True, dilation_rate=3)
    experiment_3_history = train_model(experiment_3_model, ds_train, ds_test)
    print_graphic(experiment_3_history)
    history_list.append(experiment_3_history)

    experiment_4_model = build_model(dropout='spatial', activation='swish', use_residual=False, normalization='group',
                                     use_atrous_conv=False)
    experiment_4_history = train_model(experiment_4_model, ds_train, ds_test)
    print_graphic(experiment_4_history)
    history_list.append(experiment_4_history)

    print('\n\tПередавальне навчання та fine-tuning з архітектурою VGG16 (Visual Geometry Group 16).\n')
    experiment_5_model, experiment_5_history = vgg(ds_train, ds_test)
    print_graphic(experiment_5_history)
    history_list.append(experiment_5_history)

    print('\n\tПередавальне навчання та fine-tuning з архітектурою MobileNetV2.\n')
    experiment_6_model, experiment_6_history = mobile_net(ds_train, ds_test)
    print_graphic(experiment_6_history)
    history_list.append(experiment_6_history)

    plot_all_experiments(history_list)

    create_results_table(history_list)



if __name__ == "__main__":
    main()
