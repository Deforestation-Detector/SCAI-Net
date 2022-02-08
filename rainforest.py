from ast import parse
import os
import pandas as pd
import tensorflow as tf
import scai_utils as su
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import argparse

MODEL_BATCH_SIZE = 32
EPOCHS = 3
CHECKPOINT_PATH = 'checkpoints/'
ARCH = 'ResNet50V2'
MODELS = []

def main():

    training = True

    parser = argparse.ArgumentParser()
    # Verbose mode. Display loss graphs, precision graphs, confusion matrices etc
    parser.add_argument('-v', action='store_true')
    # Training mode. Train the next N models that follow this flag.
    parser.add_argument('-t', action='store_true')
    # Load mode. Load the next N models that follow this flag from the working directory.
    parser.add_argument('-l', action='append', nargs='+')
    # Evaluate model. Evaluate the next N models that follow this flag.
    parser.add_argument('-e', action='append', nargs='+')
    # Train and save mode. Train the next N models that follow this flag and save each.
    parser.add_argument('-ts', action='store_true')

    args = parser.parse_args()
    argc = len(sys.argv)

    print(f'{args.v=}, {args.l=}, {args.e=}, {args.ts=}, {argc=}')

    if argc != 0 and sys.argv[0] == '-l':
        training = False
    train_dataframe = pd.read_csv('data/train_v2.csv').astype(str)
    train_dataframe['image_name'] += '.jpg'
    su.set_NLABELS(train_dataframe)

    mlb = MultiLabelBinarizer()
    mlb.fit(train_dataframe["tags"].str.split(" "))

    classes = [f"{c}" for c in mlb.classes_]

    ids = pd.DataFrame(mlb.fit_transform(train_dataframe['tags'].str.split(' ')), columns = classes)

    train_df = pd.concat( [train_dataframe[['image_name']], ids], axis=1 )
    train_dataframe = None

    train_dg, val_dg = su.create_data(train_df, classes)

    print(f'train_df.head =\n{train_df.head(n = 5)}')
    model = None

    if training:
        model = su.create_transfer_model(ARCH)
        su.compile_model(model)

        if os.path.isdir(CHECKPOINT_PATH) == False:
            os.mkdir(CHECKPOINT_PATH)

        val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH + ARCH + '/',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        history = model.fit(train_dg,
            epochs = EPOCHS,
            callbacks = [val_loss_checkpoint],
            batch_size = MODEL_BATCH_SIZE,
            validation_data = val_dg,
            verbose = 1
        )
        
        history_df = pd.DataFrame(history.history)
        history_df.head(n = 5)

        su.plot_history(history_df, ('Loss', 'loss'))
        su.plot_history(history_df, ('Precision', 'precision'))
        MODELS.append(('Transfer', model))
    else:
        model = tf.keras.models.load_model(CHECKPOINT_PATH + ARCH + '/')
        MODELS.append(('Transfer', model))
        precisions = su.evalModels(MODELS, val_dg)
        for model_name in precisions:
            print(f"{model_name}'s precision is {precisions[model_name]:.6f}")

    su.eyeTestPredictions(model, val_dg, classes)

    confusion_matrices = su.confusionMatrices(MODELS, val_dg)
    su.plotConfusionMatrices(confusion_matrices, classes)

# %%
if __name__ == "__main__":
    main()