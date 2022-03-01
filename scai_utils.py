import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as Kb
from tensorflow.keras.applications import ResNet50V2, Xception, VGG16, VGG19, MobileNetV2, EfficientNetV2L, EfficientNetV2M, EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Input, Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import math
from typing import TypeVar

IMG_DIMS = 256
DATA_PATH = 'data/train-jpg/'
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
THRESHOLD = 0.5
N_LABELS = None

TRANSFER_ARCHITECTURES = {
    'Xception': Xception(
        weights='imagenet', include_top=False, input_shape=(
            256, 256, 3)), 'ResNet50V2': ResNet50V2(
                weights='imagenet', include_top=False, input_shape=(
                    256, 256, 3)), 'VGG16': VGG16(
                        weights='imagenet', include_top=False, input_shape=(
                            256, 256, 3)), 'VGG19': VGG19(
                                weights='imagenet', include_top=False, input_shape=(
                                    256, 256, 3)), 'MobileNetV2': MobileNetV2(
                                        weights='imagenet', include_top=False, input_shape=(
                                            256, 256, 3)),
                                            'EfficientNetV2L': EfficientNetV2L(
                                        weights='imagenet', include_top=False, input_shape=(
                                            256, 256, 3)),
                                            'EfficientNetV2M': EfficientNetV2M(
                                        weights='imagenet', include_top=False, input_shape=(
                                            256, 256, 3)),
                                            'EfficientNetB7': EfficientNetB7(
                                        weights='imagenet', include_top=False, input_shape=(
                                            256, 256, 3)), }


def set_NLABELS(train_dataframe: pd.DataFrame) -> None:
    global N_LABELS
    curr_count = 0
    unique_labels = {}
    for line in train_dataframe['tags'].values:
        for label in line.split():
            if label not in unique_labels:
                unique_labels[label] = curr_count
                curr_count += 1

    N_LABELS = len(unique_labels)


def create_data(
        train_df: pd.DataFrame,
        classes: np.ndarray) -> "tuple[tf.keras.preprocessing.image.ImageDataGenerator]":
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        # width_shift_range = 0.15,
        # height_shift_range = 0.15,
        # channel_shift_range = 0.5
        # brightness_range = (0.2, 0.7),
        # shear_range = 0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        rescale=1 / 255,
    )

    train_dg = datagen.flow_from_dataframe(
        train_df,
        directory='./data/train-jpg/',
        x_col='image_name',
        y_col=classes,
        class_mode='raw',
        subset='training',
        validate_filenames=False,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_dg = datagen.flow_from_dataframe(
        train_df,
        directory='./data/train-jpg/',
        x_col='image_name',
        y_col=classes,
        class_mode='raw',
        subset='validation',
        validate_filenames=False,
        batch_size=VAL_BATCH_SIZE,
        shuffle=True,
    )

    print(f'{train_dg = }')

    return train_dg, val_dg


def f1(y_true: tf.constant, y_pred: tf.constant) -> tf.constant:
    y_pred = Kb.round(y_pred)
    tp = Kb.sum(Kb.cast(y_true * y_pred, 'float'), axis=0)
    tn = Kb.sum(Kb.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = Kb.sum(Kb.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = Kb.sum(Kb.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + Kb.epsilon())
    r = tp / (tp + fn + Kb.epsilon())

    f1 = 2 * p * r / (p + r + Kb.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return Kb.mean(f1)


def f1_loss(y_true: tf.constant, y_pred: tf.constant) -> tf.constant:

    y_true = tf.cast(y_true, tf.float32)

    tp = Kb.sum(Kb.cast(y_true * y_pred, 'float'), axis=0)
    tn = Kb.sum(Kb.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = Kb.sum(Kb.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = Kb.sum(Kb.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + Kb.epsilon())
    r = tp / (tp + fn + Kb.epsilon())

    f1 = 2 * p * r / (p + r + Kb.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - Kb.mean(f1)


def compile_model(model: tf.keras.Model) -> None:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        # loss=tf.keras.losses.BinaryCrossentropy(),
        loss=f1_loss,
        optimizer=opt,
        metrics=[tf.keras.metrics.Precision()]
    )


def create_transfer_model(ARCH: str) -> tf.keras.Model:
    base_model = TRANSFER_ARCHITECTURES[ARCH]
    initializer = tf.keras.initializers.HeNormal()

    # for layer in base_model.layers:
    #   layer.trainable = False

    transfer_model = Sequential()
    transfer_model.add(base_model)
    transfer_model.add(GlobalAveragePooling2D())
    transfer_model.add(BatchNormalization())

    transfer_model.add(Dense(256, kernel_initializer=initializer))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Activation('relu'))

    transfer_model.add(Dense(128, kernel_initializer=initializer))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Activation('relu'))

    transfer_model.add(Dense(N_LABELS, activation=Activation('sigmoid')))

    return transfer_model


def plot_history(history_df: pd.DataFrame, y: tf.constant) -> None:
    plt.figure(figsize=(10, 8))

    plt.title(f"{y[0]} over time")
    plt.xlabel('Epochs')
    plt.ylabel(f'{y[0]}')
    history_df[y[1]].plot(label=f'Training {y[0]}')
    history_df['val_' + y[1]].plot(label=f'Validation {y[0]}')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def reverseHot(label_numpy: np.ndarray, classes: 'list[str]') -> 'list[str]':
    label = []

    for i in label_numpy:
        label.append(classes[i])
    return ' '.join(label)


def displayGridItem(
        idx: int,
        X: tf.data.Dataset,
        y: tf.data.Dataset,
        prediction: str,
        classes: 'list[str]') -> None:
    img = tf.cast(X[idx] * 255, tf.uint8)
    # print(f"{img = }")
    indices = tf.where(y[idx] == 1).numpy()
    label_arr = []
    for index in indices:
        label_arr.append(classes[index[0]])
    label = ' '.join(label_arr)
    plt.axis('off')
    plt.imshow(img)
    plt.title(f'P: {prediction}\nL: {label}')


def eyeTestPredictions(
        model: 'tf.keras.Model',
        datagen: 'tf.keras.preprocessing.image.ImageDataGenerator',
        classes: 'list[str]') -> None:
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.8)
    rows, columns = 5, 4

    for x_batch, y_batch in datagen:
        break

    idx_array = np.random.choice(
        np.arange(TRAIN_BATCH_SIZE),
        size=20,
        replace=False)
    for iter, image_idx in enumerate(idx_array):
        y_hat_probs = model.predict(x_batch)
        prediction_hot = (y_hat_probs[image_idx] > THRESHOLD).nonzero()[0]
        prediction = reverseHot(prediction_hot, classes)
        f = fig.add_subplot(rows, columns, iter + 1)
        displayGridItem(image_idx, x_batch, y_batch, prediction, classes)

    plt.show()


def confusionMatrices(
        models: 'list[tf.keras.Model]',
        dataset: tf.data.Dataset) -> dict:
    confusion_matrices = {}

    cardinality = len(dataset) * VAL_BATCH_SIZE

    for model_name, model in models:
        percent_complete = 0
        percent_increments = 0.1
        milestone = percent_increments
        i = 0
        confusion_matrices[model_name] = np.zeros((N_LABELS, 2, 2)).astype(int)

        print(f'Obtaining confusion matrix for {model_name}')
        for features, labels in dataset:
            prob_densities = model.predict(features)
            y_hat = tf.convert_to_tensor(
                np.where(prob_densities < 0.5, 0., 1.))
            confuse_matrix = multilabel_confusion_matrix(
                labels, y_hat).astype(int)
            confusion_matrices[model_name] += confuse_matrix
            percent_complete += VAL_BATCH_SIZE / cardinality
            i += 1

            if percent_complete > milestone:
                print(f'{(percent_complete * 100):.0f}% complete')
                milestone += percent_increments
            if i > 126:
                break
        print(f'100% complete. Confusion matrix for {model_name} calculated.')

    return confusion_matrices


def ensembleConfusion(
        models: 'list[tf.keras.Model]',
        dataset: tf.data.Dataset) -> np.ndarray:
    cardinality = len(dataset) * VAL_BATCH_SIZE
    confusion_matrix = np.zeros((N_LABELS, 2, 2)).astype(int)

    i = 0
    percent_complete = 0
    percent_increments = 0.1
    milestone = percent_increments
    num_models = len(models)
    for features, labels in dataset:
        batch_size = labels.shape[0]
        prob_densities = np.zeros((batch_size, 17))

        for model in models:
            prob_densities += model.predict(features)

        prob_densities /= num_models
        y_hat = tf.convert_to_tensor(np.where(prob_densities < 0.5, 0., 1.))
        confusion_matrix += multilabel_confusion_matrix(
            labels, y_hat).astype(int)
        percent_complete += VAL_BATCH_SIZE / cardinality

        i += 1

        if percent_complete > milestone:
            print(f'{(percent_complete * 100):.0f}% complete')
            milestone += percent_increments
        if i > 126:
            break
    print(f'100% complete. Confusion matrix calculated.')

    return confusion_matrix


def plotConfusionMatrices(
        confusion_matrices: dict,
        classes: 'list[str]') -> None:
    sqt = math.sqrt(N_LABELS)
    rows, columns = math.ceil(sqt), math.floor(sqt)

    for model_name in confusion_matrices:
        fig = plt.figure(figsize=(20, 20))
        print(f"Heat map for {model_name} model")
        for i in range(N_LABELS):
            fig.add_subplot(rows, columns, i + 1)
            plt.title(f'{classes[i]}')
            sns.heatmap(
                confusion_matrices[model_name][i],
                annot=True,
                fmt='d',
            )
        plt.show()


def plotEnsembleConfusion(
        confusion_matrix: np.ndarray,
        classes: 'list[str]') -> None:
    sqt = math.sqrt(N_LABELS)
    rows, columns = math.ceil(sqt), math.floor(sqt)

    fig = plt.figure(figsize=(20, 20))
    for i in range(N_LABELS):
        fig.add_subplot(rows, columns, i + 1)
        plt.title(f'{classes[i]}')
        sns.heatmap(
            confusion_matrix[i],
            annot=True,
            fmt='d',
        )
    plt.show()
