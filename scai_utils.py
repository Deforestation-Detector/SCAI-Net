# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import tensorflow.keras.backend as Kb
from tensorflow.keras.applications import ResNet50V2, Xception, VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Input, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, load_model
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import math
# %%
IMG_DIMS = 256
DATA_PATH = 'data/train-jpg/'
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
THRESHOLD = 0.5
N_LABELS = None
MAPPING = None
# %%
def readImage(filename_tensor, resize = [IMG_DIMS, IMG_DIMS]):
    full_path = DATA_PATH + filename_tensor.decode("utf-8") + '.jpg'

    img = Image.open(full_path).convert("RGB")
    img = np.asarray(img) / 255
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, resize)

    return img

# %%
def multihot(label_tensor):
    label_string = label_tensor.decode("utf-8")
    label = tf.zeros([N_LABELS], dtype=tf.float16)
    tokens = label_string.split(' ')

    for k in range(len(tokens)):
        label += MAPPING[tokens[k]]

    return label
# %%
def symbolicRealMapping(filename_tensor, label_tensor):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    img = tf.numpy_function(readImage, [filename_tensor], tf.float32)
    label_multihot = tf.numpy_function(multihot, [label_tensor], tf.float16)

    # # print(f"{img = }")

    return img, label_multihot
# %%
# ### Defining the metric function
def f1(y_true, y_pred):
    y_pred = Kb.round(y_pred)
    tp = Kb.sum(Kb.cast(y_true*y_pred, 'float'), axis=0)
    tn = Kb.sum(Kb.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = Kb.sum(Kb.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = Kb.sum(Kb.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + Kb.epsilon())
    r = tp / (tp + fn + Kb.epsilon())

    f1 = 2*p*r / (p+r+Kb.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return Kb.mean(f1)
# %%
def f1_loss(y_true, y_pred):
    
    tp = Kb.sum(Kb.cast(y_true*y_pred, 'float'), axis=0)
    tn = Kb.sum(Kb.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = Kb.sum(Kb.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = Kb.sum(Kb.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + Kb.epsilon())
    r = tp / (tp + fn + Kb.epsilon())

    f1 = 2*p*r / (p+r+Kb.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - Kb.mean(f1)
# %%
def compile_model(model):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = opt,
        metrics = [tf.keras.metrics.Precision()]
    )
# %%
def create_model(n_labels):
    # base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
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

    transfer_model.add(Dense(n_labels, activation = Activation('sigmoid')))

    return transfer_model
# %%
def plot_history(history_df, y):
    plt.figure(figsize=(10,8))

    plt.title(f"{y[0]} over time")
    plt.xlabel('Epochs')
    plt.ylabel(f'{y[0]}')
    history_df[y[1]].plot(label=f'Training {y[0]}')
    history_df['val_' + y[1]].plot(label=f'Validation {y[0]}')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
# %%
def reverseHot(label_numpy, label2name):
    label = []
    for i in label_numpy:
        label.append(label2name[i])
    return ' '.join(label)
# %%
def displayGridItem(idx, X, y, prediction, label2name):
    img = tf.cast(X[idx] * 255, tf.uint8)
    # print(f"{img = }")
    indices = tf.where(y[idx] == 1).numpy()
    label_arr = []
    for index in indices:
        label_arr.append(label2name[index[0]])
    label = ' '.join(label_arr)
    plt.axis('off')
    plt.imshow(img)
    plt.title(prediction + '\n' + label)
# %%
def eyeTestPredictions(model, batch, label2name):
    fig = plt.figure(figsize=(20, 20))
    rows, columns = 5, 4

    idx_array = np.random.choice(np.arange(TRAIN_BATCH_SIZE), size=20, replace=False)
    for iter, image_idx in enumerate(idx_array):
        y_hat_probs = model.predict(batch[0])
        prediction_hot = (y_hat_probs[image_idx] > THRESHOLD).nonzero()[0]
        prediction = reverseHot(prediction_hot, label2name)
        f = fig.add_subplot(rows, columns, iter + 1)
        displayGridItem(image_idx, batch[0], batch[1], prediction, label2name)

    plt.show()
# %%
def evalModels(models, dataset):
    precisions = {}
    for model_name, model in models:
        print(f"evaluating {model_name}: ")
        _, model_precision = model.evaluate(dataset)
        precisions[model_name] = model_precision
    
    return precisions
# %%
def confusionMatrices(models, dataset):
    confusion_matrices = {}
    _, labels = dataset

    for model_name, model in models:
        prob_densities = model.predict(dataset)
        y_hat = tf.convert_to_tensor(np.where(prob_densities < 0.5, 0., 1.))
        confuse_matrix = multilabel_confusion_matrix(labels, y_hat)
        confusion_matrices[model_name] = confuse_matrix       
    
    return confusion_matrices
# %%
def plotConfusionMatrices(confusion_matrices, label2name, n_labels):
    sqt = math.sqrt(n_labels)
    rows, columns = math.ceil(sqt), math.floor(sqt)

    for model_name in confusion_matrices:
        fig = plt.figure(figsize=(20, 20))
        print(f"Heat map for {model_name} model")
        for i in range(n_labels):
            fig.add_subplot(rows, columns, i + 1)
            plt.title(f'{label2name[i]}')
            sns.heatmap(
                confusion_matrices[model_name][i],
                annot=True,
                fmt='d',
            )
        plt.show()