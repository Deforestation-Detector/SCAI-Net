import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as Kb
from tensorflow.keras.applications import ResNet50V2, Xception, VGG16, VGG19, MobileNetV2, InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Input, Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import math

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
                                            'InceptionResNetV2': InceptionResNetV2(
                                        weights='imagenet', include_top=False, input_shape=(
                                            256, 256, 3)), }


def set_NLABELS(train_dataframe: pd.DataFrame) -> None:
    '''Stores the number of labels in N_LABELS.

    Keyword arguments:
    train_dataframe -- the Dataframe containing images and corresponding labels 
    '''
    if isinstance(train_dataframe, pd.DataFrame) == False:
        return None, None
    df_columns = train_dataframe.columns

    # Dataset should only have 2 colums: tags and images
    if 'image_name' not in df_columns or 'tags' not in df_columns:
        return None

    if len(df_columns) != 2:
        return None

    global N_LABELS
    curr_count = 0
    unique_labels = {}
    for line in train_dataframe['tags'].values:
        if isinstance(line, str) == False:
            N_LABELS = None
            return

        # Need to separate the labels by whitespaces and look for unique labels in it
        for label in line.split():
            if label not in unique_labels:
                unique_labels[label] = curr_count
                curr_count += 1

    N_LABELS = len(unique_labels)

def create_data(
        train_df: pd.DataFrame,
        classes: np.ndarray) -> "tuple[tf.keras.preprocessing.image.ImageDataGenerator]":
    '''Returns ImageDataGenerators for training and validation.

    Keyword arguemnts:
    train_df -- the Dataframe containing training data
    classes -- MultiLabelBinarizer array corresponding to the different classes
    '''
    if isinstance(train_df, pd.DataFrame) == False or isinstance(classes, np.ndarray) == False:
        return None, None

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
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

    return train_dg, val_dg


def f1(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
    '''Returns f1 loss.

    Keyword arguments:
    y_true -- ground truth predictions
    y_pred -- predictions from model
    '''
    if tf.is_tensor(y_true) == False or tf.is_tensor(y_pred) == False:
        return None

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
    '''Returns differentiable f1 loss.

    Keyword arguments:
    y_true -- ground truth predictions
    y_pred -- predictions from model
    '''
    if tf.is_tensor(y_true) == False or tf.is_tensor(y_pred) == False:
        return None

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
    '''Initializes a tensorflow model.

    Keyword arguments:
    model -- a tensorflow model to be compiled
    '''
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=opt,
        metrics=[tf.keras.metrics.Precision()]
    )


def create_transfer_model(ARCH: str) -> tf.keras.Model:
    '''Returns a transfer learning model from the given ARCH.

    Keyword arguments:
    ARCH -- name of the desired model
    '''
    if isinstance(ARCH, str) == False or ARCH not in TRANSFER_ARCHITECTURES:
        return None

    base_model = TRANSFER_ARCHITECTURES[ARCH]
    initializer = tf.keras.initializers.HeNormal()

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

    transfer_model.add(Dense(N_LABELS, activation=Activation(activation = 'sigmoid')))

    return transfer_model


def plot_history(history_df: pd.DataFrame, y: tf.constant) -> None:
    '''Plot the model training history.

    Keyword arguments:
    history_df -- the training history Dataframe
    y -- the metric to measure
    '''
    if isinstance(history_df, pd.DataFrame) == False or tf.is_tensor(y) == False:
        return

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
    '''Returns a space delimited string of labels. 

    Keyword arguments:
    label_numpy -- the raw model prediction labels 
    classes -- the human readable classes  
    '''
    if isinstance(label_numpy, np.ndarray) == False or isinstance(classes, list) == False:
        return None

    label = []

    for i in label_numpy:
        if i > len(classes) or isinstance(classes[i], str) == False:
            return None
        label.append(classes[i])
    return ' '.join(label)


def displayGridItem(
        idx: int,
        X: tf.data.Dataset,
        y: tf.data.Dataset,
        prediction: str,
        classes: 'list[str]') -> None:
    '''Plots a single image.

    Keyword arguments:
    idx -- index of the image in X
    X -- input/features dataset
    y -- output/predictions dataset
    prediction -- the prediction made by the model
    classes -- the different classes the model can predict
    '''
    # Need to transform the images to (0, 256) then render them
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
    '''Plots images for human evaluation.

    Keyword arguments:
    model -- model
    datagen -- images the model interprets
    classes -- possible categories the model can predict
    '''
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.8)
    rows, columns = 5, 4

    # Need to grab first batch in datagenerator for testing
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
    '''Plots a confusion matrix for each model for the given dataset.

    Keyword arguments:
    models -- the models to evaluate
    dataset -- the dataset to test the models against
    '''
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
            # Need to convert the model predictions to 0, 1
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
    '''Calculates confusion matrices of each model for the given dataset.

    Keyword arguments:
    models -- the models to evaluate
    dataset -- the dataset to test the models against
    '''
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

        # Need to average predictions of all models then convert them to 0, 1
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
    '''Plots a grid of confusion matrices for each model given.

    Keyword arguments:
    confusion_matrices -- the evaluated models
    classes -- the possible categories the model can predict
    '''
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
    '''Plots a grid of confusion matrices for each model given.

    Keyword arguments:
    confusion_matrices -- the evaluated models
    classes -- the possible categories the model can predict
    '''
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