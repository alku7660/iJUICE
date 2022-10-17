import numpy as np
import pandas as pd
from address import dataset_dir
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, data_str, seed_int, train_fraction, label_str,
                 df, binary, categorical, ordinal, continuous,
                 step, ) -> None:
        self.name = data_str
        self.seed = seed_int
        self.train_fraction = train_fraction
        self.label_name = label_str
        self.df = df
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.step = step
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.df, self.df[self.label_name], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target = self.balance_train_data()
    
    def balance_train_data(self):
        """
        Method to balance the training dataset using undersampling of majority class
        """
        train_data_label = self.train_df[self.label_name]
        label_value_counts = train_data_label.value_counts()
        samples_per_class = label_value_counts.min()
        balanced_train_df = pd.concat([self.train_df[(train_data_label == 0).to_numpy()].sample(samples_per_class, random_state = self.seed),
        self.df[(train_data_label == 1).to_numpy()].sample(samples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        balanced_train_df_label = balanced_train_df[self.label_name]
        try:
            del balanced_train_df[self.label_name]
        except:
            del balanced_train_df[self.label_name[0]]
        return balanced_train_df, balanced_train_df_label