import numpy as np
import pandas as pd
import copy
from address import dataset_dir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class Dataset:

    def __init__(self, data_str, seed_int, train_fraction, label_str,
                 df, binary, categorical, ordinal, continuous,
                 step) -> None:
        self.name = data_str
        self.seed = seed_int
        self.train_fraction = train_fraction
        self.label_name = label_str
        self.df = df
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.features = self.df.columns.to_list()
        self.step = step
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.df, self.df[self.label_name], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target = self.balance_train_data()
        self.bin_enc, self.cat_enc, self.scaler = self.encoder_scaler_fit()
        self.bin_enc_cols, self.cat_enc_cols = self.encoder_scaler_cols()
        self.processed_features = self.bin_enc_cols + self.cat_enc_cols + self.ordinal + self.continuous
        self.transformed_train_df = self.transform_data(self.train_df)
        self.transformed_train_np = self.transformed_train_df.to_numpy()
        self.transformed_test_df = self.transform_data(self.test_df)
        self.transformed_test_np = self.transformed_test_df.to_numpy()
        self.train_target, self.test_target = self.change_targets_to_numpy()
        self.undesired_class = self.undesired_class_data()
        self.feat_type = self.define_feat_type()
        self.feat_mutable = self.define_feat_mutability()
        self.feat_directionality = self.define_feat_directionality()
        self.feat_step = self.define_feat_step()
        self.feat_cat = self.define_feat_cat()
        self.idx_cat_cols_dict = self.idx_cat_columns()
        self.feat_dist, self.processed_feat_dist = self.feature_distribution()

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
    
    def encoder_scaler_fit(self):
        """
        Method that fits the encoders and scaler for the dataset
        """
        bin_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        scaler = MinMaxScaler(clip=True)
        bin_enc.fit(self.train_df[self.binary])
        cat_enc.fit(self.train_df[self.categorical])
        scaler.fit(self.train_df[self.ordinal + self.continuous])
        return bin_enc, cat_enc, scaler

    def encoder_scaler_cols(self):
        """
        Method that extracts the encoded columns from the encoders
        """
        return self.bin_enc.get_feature_names_out(self.binary), self.cat_enc.get_feature_names_out(self.categorical)

    def transform_data(self, data_df):
        """
        Method that transforms the input dataframe using the encoder and scaler
        """
        data_bin, data_cat, data_ord_cont = data_df[self.binary], data_df[self.categorical], data_df[self.ordinal + self.continuous]
        enc_data_bin, enc_data_cat, sca_data_ord_cont = self.bin_enc.transform(data_bin).toarray(), self.cat_enc.transform(data_cat).toarray(), self.scaler.transform(data_ord_cont).toarray()
        enc_data_bin_df = pd.DataFrame(enc_data_bin, index=data_bin.index, columns=self.bin_enc_cols)
        enc_data_cat_df = pd.DataFrame(enc_data_cat, index=data_cat.index, columns=self.cat_enc_cols)
        sca_data_ord_cont_df = pd.DataFrame(sca_data_ord_cont, index=data_ord_cont.index, columns=self.ordinal+self.continuous)
        transformed_data_df = pd.concat((enc_data_bin_df, enc_data_cat_df, sca_data_ord_cont_df), axis=1)
        return transformed_data_df

    def undesired_class_data(self):
        """
        Method to obtain the undesired class
        """
        if self.name in ['compass','credit','german','heart','synthetic_disease']:
            undesired_class = 1
        elif self.name in ['ionosphere','adult','synthetic_athlete']:
            undesired_class = 0
        return undesired_class
    
    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        """
        feat_type = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_type.index.tolist()
        if self.name == 'ionosphere':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i:
                    feat_type.loc[i] = 'cat'
                elif i in ['Age','SleepHours']:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Smokes' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Diet' in i or 'Stress' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Weight' in i:
                    feat_type.loc[i] = 'ord'
                elif i in ['Age','ExerciseMinutes','SleepHours']:
                    feat_type.loc[i] = 'cont'
        return feat_type

    def define_feat_mutability(self):
        """
        Method that outputs mutable features per dataset
        """
        feat_mutable = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_mutable.index.tolist()
        if self.name == 'ionosphere':
            for i in feat_list:
                if i == '0':
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if i in ['Age','Sex']:
                    feat_mutable[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_mutable[i] = 1
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if i == 'Age':
                    feat_mutable[i] = 0
                elif 'Weight' in i or 'ExerciseMinutes' in i or 'SleepHours' in i or 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_mutable[i] = 1
        return feat_mutable

    def define_feat_directionality(self):
        """
        Method that outputs change directionality of features per dataset
        """
        feat_directionality = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_directionality.index.tolist()
        if self.name == 'ionosphere':
            for i in feat_list:
                feat_directionality[i] = 'any'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_directionality[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_directionality[i] = 'any'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i:
                    feat_directionality[i] = 0
                elif 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_directionality[i] = 'any'
                elif 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_directionality[i] = 'any'
        return feat_directionality

    def define_feat_step(self):
        """
        Method that estimates the step size of all features (used for ordinal features)
        """
        feat_step = pd.Series(data=1/(self.scaler.data_max_ - self.scaler.data_min_), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['ord','cont']])
        for i in self.feat_type.keys().tolist():
            if self.feat_type.loc[i] == 'cont':
                feat_step.loc[i] = self.step
            elif self.feat_type.loc[i] == 'ord':
                continue
            else:
                feat_step.loc[i] = 0
        feat_step = feat_step.reindex(index = self.feat_type.keys().to_list())
        return feat_step

    def define_feat_cat(self):
        """
        Method that assigns categorical groups to different one-hot encoded categorical features
        """
        feat_cat = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_cat.index.tolist()
        if self.name == 'ionosphere':
            for i in feat_list:
                feat_cat[i] = 'non'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'SleepHours' in i:
                    feat_cat.loc[i] = 'non'
                elif 'TrainingTime' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Sport' in i:
                    feat_cat.loc[i] = 'cat_3'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i or 'Smokes' in i or 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_cat.loc[i] = 'non'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Stress' in i:
                    feat_cat.loc[i] = 'cat_2'
        return feat_cat
    
    def idx_cat_columns(self):
        """
        Method that obtains the indices of the columns of the categorical variables 
        """
        feat_index = range(len(self.processed_features))
        dict_idx_cat = {}
        for i in self.cat_enc_cols:
            if i[:-2] not in list(dict_idx_cat.keys()): 
                cat_cols_idx = [s for s in feat_index if i[:-1] in self.processed_features[s]]
                dict_idx_cat[i[:-2]] = cat_cols_idx
        return dict_idx_cat

    def feature_distribution(self):
        """
        Method to calculate the distribution for all features
        """
        num_instances_train_df = self.train_df.shape[0]
        num_instances_processed_train = self.transformed_train_df.shape[0]
        feat_dist = {}
        processed_feat_dist = {}
        all_non_con_feat = self.binary+self.categorical+self.ordinal
        all_non_con_processed_feat = self.bin_enc_cols+self.cat_enc_cols+self.ordinal
        if len(all_non_con_feat) > 0:
            for i in all_non_con_feat:
                feat_dist[i] = ((self.train_df[i].value_counts()+1)/(num_instances_train_df+len(np.unique(self.train_df[i])))).to_dict() # +1 for laplacian counter
        if len(self.continuous) > 0:
            for i in self.continuous:
                feat_dist[i] = {'mean': self.train_df[i].mean(), 'std': self.train_df[i].std()}
                processed_feat_dist[i] = {'mean': self.transformed_train_df[i].mean(), 'std': self.transformed_train_df[i].std()}
        if len(all_non_con_processed_feat) > 0:
            for i in all_non_con_processed_feat:
                processed_feat_dist[i] = ((self.transformed_train_df[i].value_counts()+1)/(num_instances_processed_train+len(np.unique(self.transformed_train_df[i])))).to_dict() # +1 for laplacian counter
        return feat_dist, processed_feat_dist

    def change_targets_to_numpy(self):
        """
        Method that changes the targets to numpy if they are dataframes
        """
        if isinstance(self.train_target, pd.Series) or isinstance(self.train_target, pd.DataFrame):
            train_target = self.train_target.to_numpy().reshape((len(self.train_target.to_numpy()),))
        if isinstance(self.test_target, pd.Series) or isinstance(self.test_target, pd.DataFrame):
            test_target = self.test_target.to_numpy().reshape((len(self.test_target.to_numpy()),))
        return train_target, test_target

def load_dataset(data_str, train_fraction, seed, step):
    """
    Function to load all datasets according to data_str and train_fraction
    """
    if data_str == 'ionosphere':
        binary = []
        categorical = []
        ordinal = []
        continuous = ['0','2','4','5','6','7','26','30'] # Chosen based on MDI
        label = 'label'
        df = pd.read_csv(dataset_dir+'/ionosphere/processed_ionosphere.csv',index_col=0)
    elif data_str == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        ordinal = []
        continuous = ['Age','SleepHours']
        label = 'Label'
        df = pd.read_csv(dataset_dir+'synthetic_athlete/processed_synthetic_athlete.csv',index_col=0)
    elif data_str == 'synthetic_disease':
        binary = ['Smokes']
        categorical = ['Diet','Stress']
        ordinal = ['Weight']
        continuous = ['Age','ExerciseMinutes','SleepHours']
        label = 'Label'
        processed_df = pd.read_csv(dataset_dir+'synthetic_disease/processed_synthetic_disease.csv',index_col=0)

    data_obj = Dataset(data_str, seed, train_fraction, label, df,
                   binary, categorical, ordinal, continuous)
    return data_obj