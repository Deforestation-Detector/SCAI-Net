# %% [markdown]
# ### We can construct a mosaic of nearby tiles using this method: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36738

# %% [markdown]
# # Import Necessary Libraries
# ---

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
import math
from PIL import Image
from tensorflow.keras.applications import ResNet50V2, Xception

# %% [markdown]
# # Defining Constants
# ---

# %%
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 32
KERNEL_SIZE = 3
IMG_DIMS = 256
EPOCHS = 5
DATA_PATH = 'data/train-jpg/'
THRESHOLD = 0.5
CHECKPOINT_PATH = 'model/'

# %% [markdown]
# # Preprocess data
# ---

# %% [markdown]
# ### Obtain Labels

# %%
train_data = pd.read_csv('data/train_v2.csv')

curr_count = 0
unique_labels = {}
multihot = {}
for line in train_data['tags'].values:
    for label in line.split():
        if label not in unique_labels:
            unique_labels[label] = curr_count
            curr_count += 1

mapping = {}

n_labels = len(unique_labels)

for k, v in unique_labels.items():
    mapping[k] = np.zeros(n_labels, dtype=np.float16)
    mapping[k][v] = 1.0
    mapping[k] = tf.constant(mapping[k])


label2name = {v: k for k, v in unique_labels.items()}

print(label2name)

# %% [markdown]
# ### View Head of dataset

# %%
train_data.head(n = 5)

# %% [markdown]
# ### Auxiliary Function for multi-hotting

# %%
def multihot(label_tensor):
    label_string = label_tensor.decode("utf-8")
    label = tf.zeros([n_labels], dtype=tf.float16)
    tokens = label_string.split(' ')

    for k in range(len(tokens)):
        label += mapping[tokens[k]]

    return label

# %% [markdown]
# ### Auxiliary function for converting tensor filename to image

# %%
def readImage(filename_tensor, resize = [IMG_DIMS, IMG_DIMS]):
    full_path = DATA_PATH + filename_tensor.decode("utf-8") + '.jpg'

    img = Image.open(full_path).convert("RGB")
    img = np.asarray(img) / 255
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, resize)

    return img

# %% [markdown]
# ### Mapping to observe image and label from symbolic tensor

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

# %% [markdown]
# ### Create the spanning dataset

# %%
file_paths = train_data['image_name'].values
labels_strings = train_data['tags'].values
spanning_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_strings))
spanning_dataset = spanning_dataset.map(symbolicRealMapping)
spanning_dataset = spanning_dataset.prefetch(tf.data.AUTOTUNE)
dataset_length = len(spanning_dataset)

# %% [markdown]
# ### Split into a test and train set, batch each

# %%
train_length = math.floor(0.8 * dataset_length)
train_ds, val_ds = spanning_dataset.take(train_length).batch(TRAIN_BATCH_SIZE), spanning_dataset.skip(train_length).batch(TEST_BATCH_SIZE)

# %% [markdown]
# # Prepare the model
# ---

# %% [markdown]
# ### Model Architecture

# %%
# ds_model = Sequential()

# ds_model.add(Conv2D(filters = 28,
#     kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
#     input_shape = (IMG_DIMS, IMG_DIMS, 3),
#     activation='relu',
#     padding = 'Same'))
# ds_model.add(MaxPooling2D(pool_size = (2, 2)))

# ds_model.add(Conv2D(filters = 28,
#     kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
#     activation='relu'))
# ds_model.add(MaxPooling2D(pool_size = (2, 2)))

# ds_model.add(Conv2D(filters = 28,
#     kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
#     activation='relu'))
# ds_model.add(MaxPooling2D(pool_size = (2, 2)))

# ds_model.add(Flatten())

# ds_model.add(Dense(200, activation = 'relu'))
# ds_model.add(Dropout(0.2))

# ds_model.add(Dense(100, activation = 'relu'))
# ds_model.add(Dropout(0.1))

# ds_model.add(Dense(n_labels, activation = 'sigmoid'))

# %% [markdown]
# ### Save the best model
# %%
# if os.path.isdir(CHECKPOINT_PATH) == False:
#     os.mkdir(CHECKPOINT_PATH)

# val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=CHECKPOINT_PATH,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True
# )
# %% [markdown]
# ### Compile the model

# %%
# opt = K.optimizers.Adam(learning_rate=0.01)

# ds_model.compile(optimizer='adam',
#     loss = 'binary_crossentropy',
#     metrics=[tf.keras.metrics.Precision()])

# %% [markdown]
# # Train model
# ---

# %%
# ds_history = ds_model.fit(train_ds,
#     callbacks = [val_loss_checkpoint],
#     epochs = EPOCHS,
#     batch_size = MODEL_BATCH_SIZE,
#     validation_data = val_ds,
#     verbose = 1)

# %% [markdown]
# # Transfer Learning Model
# ---

# %% [markdown]
# ### TL Model architecture
# %%
base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
initializer = tf.keras.initializers.HeNormal()

# for layer in base_model.layers:
#   layer.trainable = False

xception_model = Sequential()
xception_model.add(base_model)
xception_model.add(GlobalAveragePooling2D())
xception_model.add(BatchNormalization())

xception_model.add(Dense(256, kernel_initializer=initializer))
xception_model.add(BatchNormalization())
xception_model.add(tf.keras.layers.Activation('relu'))

xception_model.add(Dense(128, kernel_initializer=initializer))
xception_model.add(BatchNormalization())
xception_model.add(tf.keras.layers.Activation('relu'))

xception_model.add(Dense(n_labels, activation = 'sigmoid'))
# %% [markdown]
# ### Compile TL model
# %%
xception_model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = [tf.keras.metrics.Precision()]
)
# %% [markdown]
# ### Fit the model
# %%
xception_history = xception_model.fit(train_ds,
    epochs = EPOCHS,
    batch_size = MODEL_BATCH_SIZE,
    validation_data = val_ds,
    verbose = 1)
# %% [markdown]
# # View results
# ---

# %% [markdown]
# ### Store the history in a dataframe

# %%
xception_history_df = pd.DataFrame(xception_history.history)
xception_history_df.head(n = 5)

# %% [markdown]
# ### Plotting function

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

# %% [markdown]
# ### Plotting loss and precision
# %%
plot_history(xception_history_df, ('Loss', 'loss'))
plot_history(xception_history_df, ('Precision', 'precision_2'))
# %% [markdown]
# ### Grabbing the first batch

# %%
batch0 = None
for batch in val_ds:
    batch0 = batch
    break

# NOTE: Batch1 is a TUPLE, not a tensor.
# It's comprised of two separate tensors, where the first
# element is the set of feature tensors of dimension 64x256x256x3
# because each batch is comprised of 64 elements, each being
# 256x256x3 images.
# The second element in the tuple is the tensor of multihot encodings

# %% [markdown]
# ### Auxiliary function to convert from labels to names

# %%
def reverseHot(label_numpy):
    label = []
    for i in label_numpy:
        label.append(label2name[i])
    return ' '.join(label)

# %% [markdown]
# ### Function to plot grid element with corresponding prediction and label

# %%
def displayGridItem(idx, X, y, prediction):
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
# %% [markdown]
# ### Function to make predictions on random images from batch 1
# %%
def eyeTestPredictions(model):
    fig = plt.figure(figsize=(20, 20))
    rows, columns = 5, 4
    idx_array = np.random.randint(TRAIN_BATCH_SIZE, size=20)
    for iter, image_idx in enumerate(idx_array):
        y_hat_probs = model.predict(batch0[0])
        prediction_hot = (y_hat_probs[image_idx] > THRESHOLD).nonzero()[0]
        prediction = reverseHot(prediction_hot)
        f = fig.add_subplot(rows, columns, iter + 1)
        displayGridItem(image_idx, batch0[0], batch0[1], prediction)

    plt.show()

# %%
eyeTestPredictions(xception_model)
# %%
