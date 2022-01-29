# %% [markdown]
# ### We can construct a mosaic of nearby tiles using this method: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36738

# %% [markdown]
# # Import Necessary Libraries

# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential, Model
import math
from PIL import Image
print()

# %% [markdown]
# # Defining Constants

# %%
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 32
KERNEL_SIZE = 3
IMG_DIMS = 256
EPOCHS = 10
DATA_PATH = 'data/train-jpg/'
THRESHOLD = 0.5

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
train_data.head(n = 11)

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
# ### Function to grab the label from the filename

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
# first pass, construct a list of image strips

file_paths = train_data['image_name'].values
labels_strings = train_data['tags'].values
spanning_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_strings))
spanning_dataset = spanning_dataset.map(symbolicRealMapping)
spanning_dataset = spanning_dataset.prefetch(tf.data.AUTOTUNE)
dataset_length = len(spanning_dataset)
print(f"{dataset_length}")

# %% [markdown]
# ### Split into a test and train set, batch each

# %%
train_length = math.floor(0.8 * dataset_length)
train_ds, val_ds = spanning_dataset.take(train_length).batch(TRAIN_BATCH_SIZE), spanning_dataset.skip(train_length).batch(TEST_BATCH_SIZE)

# %% [markdown]
# ### Display a target image

# %%
def show_image(idx, X, y):
    img = tf.cast(X[idx] * 255, tf.uint8)
    # print(f"{img = }")
    indices = tf.where(y[idx] == 1).numpy()
    label_arr = []
    for index in indices:
        label_arr.append(label2name[index[0]])
    label = ' '.join(label_arr)
    plt.imshow(img)
    plt.title(label)
    plt.show()

# %%
batch = None

for e in train_ds:
    batch = e
    break

show_image(24, batch[0], batch[1])

# %% [markdown]
# # Prepare the model
# ---

# %% [markdown]
# ### Define evaluation function

# %%
def macro_f1(y, y_hat):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, THRESHOLD), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

# %% [markdown]
# ### Model Architecture

# %%
ds_model = Sequential()

ds_model.add(Conv2D(filters = 28,
    kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
    input_shape = (IMG_DIMS, IMG_DIMS, 3),
    activation='relu',
    padding = 'Same'))
ds_model.add(MaxPooling2D(pool_size = (2, 2)))

ds_model.add(Conv2D(filters = 28,
    kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
    activation='relu'))
ds_model.add(MaxPooling2D(pool_size = (2, 2)))

ds_model.add(Conv2D(filters = 28,
    kernel_size = (KERNEL_SIZE, KERNEL_SIZE),
    activation='relu'))
ds_model.add(MaxPooling2D(pool_size = (2, 2)))

ds_model.add(Flatten())

ds_model.add(Dense(200, activation = 'relu'))
ds_model.add(Dropout(0.2))

ds_model.add(Dense(100, activation = 'relu'))
ds_model.add(Dropout(0.1))

ds_model.add(Dense(n_labels, activation = 'sigmoid'))

# %% [markdown]
# ### Compile the model

# %%
opt = K.optimizers.Adam(learning_rate=0.01)

ds_model.compile(optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=macro_f1)

# %% [markdown]
# # Train model
# ---

# %%
ds_history = ds_model.fit(train_ds,
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
history_df = pd.DataFrame(ds_history.history)
history_df

# %% [markdown]
# ### Plotting the loss

# %%
plt.figure(figsize=(10,8))
plt.title("Loss over time")
plt.xlabel('Epochs')
plt.ylabel('Loss')
history_df['loss'].plot(label='Training Loss')
history_df['val_loss'].plot(label='Validation loss')
plt.grid()
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### Grabbing the first batch

# %%
batch0 = None
for batch in train_ds:
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

# %%
image_idx = 31
y_hat_probs = ds_model.predict(batch0[0])
prediction_hot = (y_hat_probs[image_idx] > THRESHOLD).nonzero()[0]
prediction = reverseHot(prediction_hot)
print(f'{prediction = }')
show_image(image_idx, batch0[0], batch0[1])

# %% [markdown]
# # Exporting Model weights

# %%
ds_model.save("current_weights")


