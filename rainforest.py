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
ARCH = 'ResNet50V2/'
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
    label2name = [k for k in unique_labels]

    mlb = MultiLabelBinarizer()
    mlb.fit(train_dataframe["tags"].str.split(" "))

    new_columns = [f"{c}" for c in mlb.classes_]

    ids = pd.DataFrame(mlb.fit_transform(train_dataframe['tags'].str.split(' ')), columns = new_columns)

    train_df = pd.concat( [train_dataframe[['image_name']], ids], axis=1 )
    train_dataframe = None
    train_dg, val_dg = None, None

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 45,
        width_shift_range = 0.15,
        height_shift_range = 0.15,
        # channel_shift_range = 0.5
        # brightness_range = (0.2, 0.7),
        # shear_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = 0.2,
        rescale = 1/255
    )

    train_dg = datagen.flow_from_dataframe(
        train_df,
        directory = './data/train-jpg/',
        x_col = 'image_name',
        y_col = new_columns,
        class_mode = 'raw',
        subset = 'training',
        validate_filenames = False,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
    )

    val_dg = datagen.flow_from_dataframe(
        train_df,
        directory = './data/train-jpg/',
        # class_mode = 'multi_output',
        x_col = 'image_name',
        y_col = new_columns,
        class_mode = 'raw',
        subset = 'validation',
        validate_filenames = False,
        batch_size = VAL_BATCH_SIZE,
        shuffle = True,
    )

    print(f'train_df.head =\n{train_df.head(n = 5)}')

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

        # transfer_history = transfer_model.fit(train_datagen,
        #     epochs = EPOCHS,
        #     callbacks = [val_loss_checkpoint],
        #     batch_size = MODEL_BATCH_SIZE,
        #     validation_data = val_datagen,
        #     verbose = 1
        # )

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
    else:
        transfer_model = tf.keras.models.load_model(CHECKPOINT_PATH + ARCH)
    MODELS.append(('Transfer', transfer_model))

    confusion_matrices = su.confusionMatrices(MODELS, val_dg)
    su.plotConfusionMatrices(confusion_matrices, label2name, n_labels)

# %%
if __name__ == "__main__":
    main(sys.argv[1:])