import scai_utils as su
import pandas as pd
import unittest
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np

class SCAITestMethods(unittest.TestCase):
    def test_setNLabels(self):
        IMAGE_NAMES = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6']

        d_empty = {'image_name': [], 'tags': []}
        expected_num_labels_unique = 0
        df_empty = pd.DataFrame(data=d_empty)
        su.set_NLABELS(df_empty)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_non_unique_labels = {'image_name': IMAGE_NAMES, 'tags': ['label1 label2', 'label2 label1', 'label3 label5', 'label1 label2', 'label4 label3', 'label5 label1']}
        df_non_unique_labels = pd.DataFrame(data=d_non_unique_labels)
        expected_num_labels_non_unique = 5
        su.set_NLABELS(df_non_unique_labels)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_non_unique)
        su.N_LABELS = None
        
        d_unique_labels = {'image_name': IMAGE_NAMES, 'tags': ['label1', 'label2', 'label3', 'label4', 'label5', 'label6']}
        expected_num_labels_unique = 6
        df_unique_labels = pd.DataFrame(data=d_unique_labels)
        su.set_NLABELS(df_unique_labels)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_none = {'image_name': IMAGE_NAMES, 'tags': [None, None, None, None, None, None]}
        expected_num_labels_unique = None
        dfnone = pd.DataFrame(data=d_none)
        su.set_NLABELS(dfnone)
        num_labels_observed = su.N_LABELS
        self.assertEqual(num_labels_observed, expected_num_labels_unique)
        su.N_LABELS = None

        d_improper_column_names = {'other_images': IMAGE_NAMES, 'other_tags': ['label1 label2', 'label2 label1', 'label3 label5', 'label1 label2', 'label4 label3', 'label5 label1']}
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

        y_true_invalid = [0.8, 0.2, 0.8]
        y_pred_invalid = [0.8, 0.2, 0.8]
        f1_score_expected = None
        f1_score_invalid = su.f1(y_true_invalid, y_pred_invalid)
        self.assertEqual(f1_score_expected, f1_score_invalid)

    def test_f1Loss(self):
        y_true_equal = tf.constant([0.8, 0.2, 0.8])
        y_pred_equal = tf.constant([0.4, 0.1, 0.7])
        f1Loss_score_expected = np.around(np.float32(0.4), 3)
        f1Loss_score_equal = np.around(su.f1_loss(y_true_equal, y_pred_equal), 3)
        self.assertEqual(f1Loss_score_expected, f1Loss_score_equal)

        y_true_inequal = tf.constant([0.9, 0.8, 0.4])
        y_pred_inequal = tf.constant([0.1, 0.1, 0.6])
        f1Loss_score_inequal = np.around(su.f1_loss(y_true_inequal, y_pred_inequal), 3)
        f1Loss_score_expected = np.around(np.float32(0.717), 3)
        self.assertEqual(f1Loss_score_inequal, f1Loss_score_expected)

        y_true_invalid = [0.8, 0.2, 0.8]
        y_pred_invalid = [0.8, 0.2, 0.8]
        f1Loss_score_expected = None
        f1Loss_score_invalid = su.f1_loss(y_true_invalid, y_pred_invalid)
        self.assertEqual(f1Loss_score_expected, f1Loss_score_invalid)

    def test_createTransferModel(self):
        su.N_LABELS = 1

        ARCH_IN_ARCHITECTURES = 'MobileNetV2'
        expected_transfer_model = su.create_transfer_model(ARCH_IN_ARCHITECTURES)
        self.assertNotEqual(expected_transfer_model, None)

        ARCH_NOT_IN_ARCHITECTURES = 'MobileNetV3'
        expected_transfer_model = su.create_transfer_model(ARCH_NOT_IN_ARCHITECTURES)
        self.assertEqual(expected_transfer_model, None)

        INVALID_ARCH = None
        expected_transfer_model = su.create_transfer_model(INVALID_ARCH)
        self.assertEqual(expected_transfer_model, None)

        su.N_LABELS = None
    
    def test_reverseHot(self):
        label_valid = np.array([0, 2])
        classes_valid = ['label1', 'label2', 'label3']
        reverse_hot_valid = su.reverseHot(label_valid, classes_valid)
        reverse_hot_expected = 'label1 label3'
        self.assertEqual(reverse_hot_valid, reverse_hot_expected)

        label_invalid = [0, 2]
        classes_valid = ['label1', 'label2', 'label3']
        reverse_hot_invalid = su.reverseHot(label_invalid, classes_valid)
        reverse_hot_expected = None
        self.assertEqual(reverse_hot_invalid, reverse_hot_expected)

        label_valid = np.array([0, 2])
        classes_invalid = 'label'
        reverse_hot_invalid = su.reverseHot(label_valid, classes_invalid)
        reverse_hot_expected = None
        self.assertEqual(reverse_hot_invalid, reverse_hot_expected)

        label_valid = np.array([0, 5])
        classes_invalid = ['label1', 'label2', 'label3']
        reverse_hot_invalid = su.reverseHot(label_valid, classes_invalid)
        reverse_hot_expected = None
        self.assertEqual(reverse_hot_invalid, reverse_hot_expected)
        
        label_valid = np.array([0, 1])
        classes_invalid = ['label1', 1, 'label3']
        reverse_hot_invalid = su.reverseHot(label_valid, classes_invalid)
        reverse_hot_expected = None
        self.assertEqual(reverse_hot_invalid, reverse_hot_expected)


if __name__ == '__main__':
    unittest.main()

