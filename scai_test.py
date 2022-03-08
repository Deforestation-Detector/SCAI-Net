import scai_utils as su
import pandas as pd
import unittest
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np

class SCAITestMethods(unittest.TestCase):
    def test_setNLabels(self):
        IMAGES = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6']

        d_empty = {'images': [], 'tags': []}
        expected_num_labels_unique = 0
        df_empty = pd.DataFrame(data=d_empty)
        su.set_NLABELS(df_empty)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_non_unique_labels = {'images': IMAGES, 'tags': ['label1 label2', 'label2 label1', 'label3 label5', 'label1 label2', 'label4 label3', 'label5 label1']}
        df_non_unique_labels = pd.DataFrame(data=d_non_unique_labels)
        expected_num_labels_non_unique = 5
        su.set_NLABELS(df_non_unique_labels)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_non_unique)
        su.N_LABELS = None
        
        d_unique_labels = {'images': IMAGES, 'tags': ['label1', 'label2', 'label3', 'label4', 'label5', 'label6']}
        expected_num_labels_unique = 6
        df_unique_labels = pd.DataFrame(data=d_unique_labels)
        su.set_NLABELS(df_unique_labels)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_none = {'images': IMAGES, 'tags': [None, None, None, None, None, None]}
        expected_num_labels_unique = None
        dfnone = pd.DataFrame(data=d_none)
        su.set_NLABELS(dfnone)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_improper_column_names = {'other_images': IMAGES, 'other_tags': ['label1 label2', 'label2 label1', 'label3 label5', 'label1 label2', 'label4 label3', 'label5 label1']}
        expected_num_labels_improper_column_names = None
        df_improper_column_names = pd.DataFrame(data=d_improper_column_names)
        su.set_NLABELS(df_improper_column_names)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_improper_column_names)
        su.N_LABELS = None

    def test_createData(self):
        train_dataframe = pd.read_csv('data/MINE/MINE.csv').astype(str)
        train_dataframe['image_name'] += '.jpg'
        su.set_NLABELS(train_dataframe)

        mlb = MultiLabelBinarizer()
        mlb.fit(train_dataframe["tags"].str.split(" "))
        classes = mlb.classes_

        multihot_df = pd.DataFrame(
            mlb.fit_transform(
                train_dataframe['tags'].str.split(' ')),
            columns=classes)

        train_df = pd.concat([train_dataframe[['image_name']], multihot_df], axis=1)
        train_dataframe = None

        train_dg, val_dg = su.create_data(train_df, classes)
        train_dg_is_none = train_dg == None
        val_dg_is_none = val_dg == None

        self.assertEqual(train_dg_is_none, False)
        self.assertEqual(val_dg_is_none, False)

        train_df_none = None
        train_dg_none, val_dg_none = su.create_data(train_df_none, classes)
        self.assertEqual(train_dg_none, None)
        self.assertEqual(val_dg_none, None)

        classes_none = None
        train_dg_none, val_dg_none = su.create_data(train_df, classes_none)
        self.assertEqual(train_dg_none, None)
        self.assertEqual(val_dg_none, None)

        train_df_none, classes_none = None, None
        train_dg_none, val_dg_none = su.create_data(train_df_none, classes_none)
        self.assertEqual(train_dg_none, None)
        self.assertEqual(val_dg_none, None)
    
    def test_f1(self):
        y_true_equal = tf.constant([0.8, 0.2, 0.8])
        y_pred_equal = tf.constant([0.4, 0.1, 0.7])
        f1_score_expected = np.around(np.float32(0.571), 3)
        f1_score_equal = np.around(su.f1(y_true_equal, y_pred_equal), 3)

        self.assertEqual(f1_score_expected, f1_score_equal)

        y_true_inequal = tf.constant([0.9, 0.8, 0.4])
        y_pred_inequal = tf.constant([0.1, 0.1, 0.6])
        f1_score_inequal = np.around(su.f1(y_true_inequal, y_pred_inequal), 3)
        f1_score_expected = np.around(np.float32(0.258), 3)

        self.assertEqual(f1_score_inequal, f1_score_expected)

if __name__ == '__main__':
    unittest.main()

