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


def main():

    model_choices = ['Xception', 'ResNet50V2', 'VGG16', 'VGG19', 'MobileNetV2']
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true',
                        help='Verbose mode. Display loss graphs, precision'
                        ' graphs, confusion matrices etc')
    parser.add_argument('-t', action='append', nargs='+', type=str,
                        choices=model_choices,
                        help='Training mode. Train the next N models that'
                        ' follow this flag.')
    parser.add_argument('-l', action='append', nargs='+', type=str,
                        choices=model_choices,
                        help='Load mode. Load the next N models that follow'
                        ' this flag from the working directory.')
    parser.add_argument('-e', action='append', nargs='+', type=str,
                        choices=model_choices,
                        help='Evaluate model. Evaluate the next N models that'
                        ' follow this flag.')
    parser.add_argument(
        '-ts',
        action='append',
        nargs='+',
        type=str,
        choices=model_choices,
        help='Train and save mode. Train the next N models that'
        ' follow this flag and save each.')
    parser.add_argument(
        '-le',
        action='append',
        nargs='+',
        type=str,
        choices=model_choices,
        help='Load and evaluate mode. Load and evaluate the next N models that'
        ' follow this flag.')

    # get the number of arguments and construct the argument namespace object
    argc = len(sys.argv)
    args = parser.parse_args()
    arg_dict = vars(args)

    # initialize model dictionary, this wills store booleans for which
    # operations will be performed on each model
    model_dict = dict()

    # parse command line arguments
    if argc > 1:
        # create dictionary storing model functionality when the script is run
        for arg in arg_dict:
            # ignore arguments that do not switch on functionality for model
            # add them to this list
            if arg in ['v']:
                continue
            # for each list argument, if the model was listed, set the
            # corresponding value to True

            if arg_dict[arg] is not None:
                for model_name in arg_dict[arg].pop():
                    if model_name not in model_dict:
                        model_dict[model_name] = dict()
                    for flag in arg:
                        model_dict[model_name][flag] = True

    if args.v:
        print(f'{model_dict=}')  # should be printed during verbose mode

    # build dataframe containing training data
    train_dataframe = pd.read_csv('data/train_v2.csv').astype(str)
    train_dataframe['image_name'] += '.jpg'
    su.set_NLABELS(train_dataframe)

    # since our task is multilabel classification, we must construct a multi
    # label binarizer to binarize each of the different labels
    mlb = MultiLabelBinarizer()
    mlb.fit(train_dataframe["tags"].str.split(" "))
    classes = mlb.classes_

    if args.v:
        print(f'{type(classes) = }')  # should be printed during verbose mode

    ids = pd.DataFrame(
        mlb.fit_transform(
            train_dataframe['tags'].str.split(' ')),
        columns=classes)

    train_df = pd.concat([train_dataframe[['image_name']], ids], axis=1)
    train_dataframe = None

    train_dg, val_dg = su.create_data(train_df, classes)
    model_list = []

    if args.v:
        # should be printed during verbose mode
        print(f'train_df.head =\n{train_df.head(n = 5)}')

    # compute operations on models specified at the command line
    for model_name in model_dict:
        if args.v:
            print(f'Current architecture: {model_name}')
        is_training, is_loaded, is_evaluated, is_training = False, False, False, False

        # parse model_dict booleans for the current model
        for operation in model_dict[model_name]:
            if operation == 't':
                is_training = True
            if operation == 'e':
                is_evaluated = True
            if operation == 's':
                is_saved = True
            if operation == 'l':
                is_loaded = True

        # initialize model list. Used for model evaluation

        if is_training:
            # initialize the transfer learning model
            model = su.create_transfer_model(model_name)
            su.compile_model(model)

            # create the directory where the model checkpoint will be saved
            if os.path.isdir(CHECKPOINT_PATH) == False:
                os.mkdir(CHECKPOINT_PATH)

            callbacks = []
            if is_saved:
                # construct model checkpoint object. during training, this saves
                # the model weights at when the current weights are better than
                # the previously saved weights. optimal weights at the current
                # timestep are evaluated using validation loss.
                val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=CHECKPOINT_PATH + model_name + '/',
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                )
                callbacks = [val_loss_checkpoint]

            # train model on dataset and use the checkpoint object
            history = model.fit(train_dg,
                                epochs=EPOCHS,
                                callbacks=callbacks,
                                batch_size=MODEL_BATCH_SIZE,
                                validation_data=val_dg,
                                verbose=1
                                )

            # plot the model training history
            if is_evaluated:
                history_df = pd.DataFrame(history.history)
                history_df.head(n=5)
                su.plot_history(history_df, ('Loss', 'loss'))
                su.plot_history(history_df, ('Precision', 'precision'))
            model_list.append(model)
        elif is_loaded:
            model = tf.keras.models.load_model(
                CHECKPOINT_PATH + model_name + '/',
                compile=False)
            model_list.append(model)

            if is_evaluated:
                if args.v:
                    model.evaluate(val_dg)

                su.eyeTestPredictions(model, val_dg, classes)

    # confusion_matrices = su.confusionMatrices(model_list, val_dg)
    print(f'{model_list = }')

    confusion_matrix = su.ensembleConfusion(model_list, val_dg)
    su.plotEnsembleConfusion(confusion_matrix, classes)
    # su.plotConfusionMatrices(confusion_matrices, classes)


# %%
if __name__ == "__main__":
    main()
