import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import scai_utils as su
import sys

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 32
EPOCHS = 3
CHECKPOINT_PATH = 'checkpoints/'
ARCH = 'ResNet50V2/'
MODELS = []

def main(argv):
    training = True
    argc = len(argv)

    if argc != 0 and argv[0] == '-l':
        training = False
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

    su.MAPPING, su.N_LABELS = mapping, n_labels
    label2name = [k for k in unique_labels]

    train_data.head(n = 5)

    file_paths = train_data['image_name'].values
    labels_strings = train_data['tags'].values
    spanning_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_strings))
    spanning_dataset = spanning_dataset.map(su.symbolicRealMapping)
    spanning_dataset = spanning_dataset.prefetch(tf.data.AUTOTUNE)
    dataset_length = len(spanning_dataset)

    train_length = math.floor(0.8 * dataset_length)
    train_ds, val_ds = spanning_dataset.take(train_length).batch(TRAIN_BATCH_SIZE), spanning_dataset.skip(train_length).batch(TEST_BATCH_SIZE)
    transfer_model = None

    if training:
        transfer_model = su.create_model(n_labels)
        su.compile_model(transfer_model)

        if os.path.isdir(CHECKPOINT_PATH) == False:
            os.mkdir(CHECKPOINT_PATH)

        val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH + ARCH,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        transfer_history = transfer_model.fit(train_ds,
            epochs = EPOCHS,
            callbacks = [val_loss_checkpoint],
            batch_size = MODEL_BATCH_SIZE,
            validation_data = val_ds,
            verbose = 1)
        
        transfer_history_df = pd.DataFrame(transfer_history.history)
        transfer_history_df.head(n = 5)

        su.plot_history(transfer_history_df, ('Loss', 'loss'))
        su.plot_history(transfer_history_df, ('Precision', 'precision'))
    else:
        transfer_model = tf.keras.models.load_model(CHECKPOINT_PATH + ARCH)
    MODELS.append(('Transfer', transfer_model))

    batch0 = None
    for batch in val_ds:
        batch0 = batch
        break

    print("Transfer model eyetest")
    su.eyeTestPredictions(transfer_model, batch0, label2name)

    precisions = su.evalModels(MODELS, val_ds)
    for model_name in precisions:
        print(f"{model_name}'s precision is {precisions[model_name]:.6f}")

    confusion_matrices = su.confusionMatrices(MODELS, batch0)

    su.plotConfusionMatrices(confusion_matrices, label2name, n_labels)

    # transfer_model.predict(val_ds)

# %%
if __name__ == "__main__":
    main(sys.argv[1:])