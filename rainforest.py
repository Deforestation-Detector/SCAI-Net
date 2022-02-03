# %% [markdown]
# ### We can construct a mosaic of nearby tiles using this method: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36738

# %% [markdown]
# # Import Necessary Libraries
# ---

# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import scai_utils as su
# %% [markdown]
# # Defining Constants
# ---

# %%
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 32
EPOCHS = 3
CHECKPOINT_PATH = 'checkpoints/'
ARCH = 'VGG16/'
# %% [markdown]
# # Preprocess data
# ---

# %% [markdown]
# ### Obtain Labels

# %%
train_data = pd.read_csv('data/train_v2.csv')

curr_count = 0
unique_labels = {}
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


label2name = [k for k in unique_labels]

print(label2name)

# %% [markdown]
# ### View Head of dataset

# %%
train_data.head(n = 5)
# %% [markdown]
# ### Create the spanning dataset

# %%
file_paths = train_data['image_name'].values
labels_strings = train_data['tags'].values
spanning_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_strings))
spanning_dataset = spanning_dataset.map(su.symbolicRealMapping)
spanning_dataset = spanning_dataset.prefetch(tf.data.AUTOTUNE)
dataset_length = len(spanning_dataset)

# %% [markdown]
# ### Split into a test and train set, batch each

# %%
train_length = math.floor(0.8 * dataset_length)
train_ds, val_ds = spanning_dataset.take(train_length).batch(TRAIN_BATCH_SIZE), spanning_dataset.skip(train_length).batch(TEST_BATCH_SIZE)

# %% [markdown]
# # Transfer Learning Model
# ---
# %%
transfer_model = su.create_model(n_labels)
# %% [markdown]
# ### Compile TL model
# %%
su.compile_model(transfer_model)
# %% [markdown]
# ### Save the best model

# %%
if os.path.isdir(CHECKPOINT_PATH) == False:
    os.mkdir(CHECKPOINT_PATH)

val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH + ARCH,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
)
# %% [markdown]
# ### Fit the model
# %%
transfer_history = transfer_model.fit(train_ds,
    epochs = EPOCHS,
    callbacks = [val_loss_checkpoint],
    batch_size = MODEL_BATCH_SIZE,
    validation_data = val_ds,
    verbose = 1)
# %% [markdown]
# # View results
# ---

# %% [markdown]
# ### Store the history in a dataframe

# %%
transfer_history_df = pd.DataFrame(transfer_history.history)
transfer_history_df.head(n = 5)
# %%
su.plot_history(transfer_history_df, ('Loss', 'loss'))
su.plot_history(transfer_history_df, ('Precision', 'precision'))
# %% [markdown]
# ### Load the saved model

# %%
saved_model = su.create_model(n_labels)
saved_model.load_weights(CHECKPOINT_PATH + ARCH)
su.compile_model(saved_model)
# %% [markdown]
# ### Grabbing the first batch

# %%
batch0 = None
for batch in val_ds:
    batch0 = batch
    break

# %% [markdown]
# ### Eye test the transfer network

# %%
print("Transfer model eyetest")
su.eyeTestPredictions(transfer_model, batch0)

# %% [markdown]
# ### Eye test the saved network

# %%
su.eyeTestPredictions(saved_model, batch0)
# %% [markdown]
# ### Define the models array and evaluate them

# %%
models = [
    ('Saved', saved_model),
    ('Transfer', transfer_model)
]
precisions = su.evalModels(models, val_ds)

for model_name in precisions:
    print(f"{model_name}'s precision is {precisions[model_name]:.6f}")

# %% [markdown]
# ### Determine the confusion
# %%
confusion_matrices = su.confusionMatrices(models, val_ds)
# %% [markdown]
# ### Plot the confusion matrices
# %%
su.plotConfusionMatrices(confusion_matrices, label2name, n_labels)
# %%
transfer_model.predict(val_ds)
# %%
saved_model.save("current_weights")