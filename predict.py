import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import scai_utils as su
seed = 420
tf.random.set_seed(seed)
np.random.seed(seed)

CLASSES = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
       'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',  
       'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
       'selective_logging', 'slash_burn', 'water']

xception_model = tf.keras.models.load_model('checkpoints/Xception')
resnet50v2_model = tf.keras.models.load_model('checkpoints/ResNet50V2')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1 / 255,
)

train_dataframe = pd.read_csv('data/MINE/MINE.csv').astype(str)
train_dataframe['image_name'] += '.jpg'
su.set_NLABELS(train_dataframe)

mlb = MultiLabelBinarizer()
binarized_df = mlb.fit(train_dataframe["tags"].str.split(" "))
classes = mlb.classes_

print(f'{type(classes) = }')

multihot_df = pd.DataFrame(
    mlb.fit_transform(
        train_dataframe['tags'].str.split(' ')),
    columns=classes)

train_df = pd.concat([train_dataframe[['image_name']], multihot_df], axis=1)

print(f'{train_dataframe = }')

dataset = datagen.flow_from_dataframe(
    train_df,
    directory='./data/MINE/',
    x_col='image_name',
    y_col=classes,
    class_mode='raw',
    subset='training',
    validate_filenames=False,
    shuffle = False,
    batch_size = 9
)

predictions = []
y_hats_xception = xception_model.predict(dataset)
y_hats_resnet50v2 = resnet50v2_model.predict(dataset)
y_hat_net = (y_hats_xception + y_hats_resnet50v2) / 2

for image_idx in range(9):
    predictions_net = (y_hat_net[image_idx] > 0.5).nonzero()[0]

    prediction = su.reverseHot(predictions_net, CLASSES)
    predictions.append(prediction)

for i, e in enumerate(predictions):
    print(f'{i} {e}')