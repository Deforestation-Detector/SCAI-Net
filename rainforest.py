import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import scai_utils as su
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import gc

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 32
EPOCHS = 3
CHECKPOINT_PATH = 'checkpoints/'
ARCH = 'ResNet50V2'
MODELS = []
DATA_PATH = 'data/train-jpg/'

def main(argv):
    training = True
    argc = len(argv)

    if argc != 0 and argv[0] == '-l':
        training = False
    train_dataframe = pd.read_csv('data/train_v2.csv').astype(str)
    train_dataframe['image_name'] += '.jpg'

    curr_count = 0
    unique_labels = {}
    for line in train_dataframe['tags'].values:
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

    mlb = MultiLabelBinarizer()
    mlb.fit(train_dataframe["tags"].str.split(" "))

    classes = [f"{c}" for c in mlb.classes_]

    ids = pd.DataFrame(mlb.fit_transform(train_dataframe['tags'].str.split(' ')), columns = classes)

    train_df = pd.concat( [train_dataframe[['image_name']], ids], axis=1 )
    train_dataframe = None

    train_dg, val_dg = su.create_data(train_df, classes)

    print(f'train_df.head =\n{train_df.head(n = 5)}')

    if training:
        transfer_model = su.create_model(ARCH, n_labels)
        su.compile_model(transfer_model)

        if os.path.isdir(CHECKPOINT_PATH) == False:
            os.mkdir(CHECKPOINT_PATH)

        val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH + ARCH + '/',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        transfer_history = transfer_model.fit(train_dg,
            epochs = EPOCHS,
            callbacks = [val_loss_checkpoint],
            batch_size = MODEL_BATCH_SIZE,
            validation_data = val_dg,
            verbose = 1
        )
        
        transfer_history_df = pd.DataFrame(transfer_history.history)
        transfer_history_df.head(n = 5)

        su.plot_history(transfer_history_df, ('Loss', 'loss'))
        su.plot_history(transfer_history_df, ('Precision', 'precision'))
        MODELS.append(('Transfer', transfer_model))
    else:
        transfer_model = tf.keras.models.load_model(CHECKPOINT_PATH + ARCH + '/')
        MODELS.append(('Transfer', transfer_model))
        precisions = su.evalModels(MODELS, val_dg)
        for model_name in precisions:
            print(f"{model_name}'s precision is {precisions[model_name]:.6f}")

    print("Transfer model eyetest")
    su.eyeTestPredictions(transfer_model, val_dg, classes)

    confusion_matrices = su.confusionMatrices(MODELS, val_dg)
    su.plotConfusionMatrices(confusion_matrices, classes, n_labels)

# %%
if __name__ == "__main__":
    main(sys.argv[1:])