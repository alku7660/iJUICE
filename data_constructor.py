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
                 step, attributes) -> None:
        self.name = data_str
        self.seed = seed_int
        self.train_fraction = train_fraction
        self.label_name = label_str
        self.df = df
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.attributes_long = attributes
        self.features = self.df.columns.to_list()
        self.step = step
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.df, self.df[self.label_name], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target = self.balance_train_data()
        self.bin_enc, self.cat_enc, self.scaler = self.encoder_scaler_fit()
        self.bin_enc_cols, self.cat_enc_cols = self.encoder_scaler_cols()
        self.processed_features = list(self.bin_enc_cols) + list(self.cat_enc_cols) + self.ordinal + self.continuous
        self.transformed_train_df = self.transform_data(self.train_df)
        self.transformed_train_np = self.transformed_train_df.to_numpy()
        self.transformed_test_df = self.transform_data(self.test_df)
        self.transformed_test_np = self.transformed_test_df.to_numpy()
        self.train_target, self.test_target = self.change_targets_to_numpy()
        self.undesired_class = self.undesired_class_data()
        self.feat_type = self.define_feat_type()
        self.feat_mutable = self.define_feat_mutability()
        self.immutables = self.get_immutables()
        self.feat_dir = self.define_feat_directionality()
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
        self.train_df[(train_data_label == 1).to_numpy()].sample(samples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
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
        enc_data_bin = self.bin_enc.transform(data_bin).toarray()
        enc_data_cat = self.cat_enc.transform(data_cat).toarray()
        sca_data_ord_cont = self.scaler.transform(data_ord_cont)
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
    
    def undesired_test(self, model):
        """
        Method to obtain the test subset with predicted undesired class
        """
        self.undesired_transformed_test_df = self.transformed_test_df.loc[model.model.predict(self.transformed_test_df) == self.undesired_class]
        self.undesired_transformed_test_np = self.undesired_transformed_test_df.to_numpy()

    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        """
        feat_type = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_type.loc[i] = 'num-ord'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_type.loc[i] = 'num-con'
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
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Age' in i or 'Native' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'ionosphere':
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
    
    def get_immutables(self):
        """
        Outputs the immutable features list according to the mutability property
        """
        immutables = []
        for i in self.feat_mutable.keys():
            if self.feat_mutable[i] == 0:
                immutables.append(i)
        return immutables

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
        all_non_con_feat = self.binary + self.categorical + self.ordinal
        all_non_con_processed_feat = list(self.bin_enc_cols) + list(self.cat_enc_cols) + self.ordinal
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

    def inverse(self, normal_x):
        """
        Method that transforms an instance back into the original space
        """
        normal_x_df = pd.DataFrame(data=normal_x, columns=self.processed_features)
        normal_x_df_bin, normal_x_df_cat, normal_x_df_ord_cont = normal_x_df[self.bin_enc_cols], normal_x_df[self.cat_enc_cols], normal_x_df[self.ordinal+self.continuous]
        x_bin = self.bin_enc.inverse_transform(normal_x_df_bin).toarray()
        x_cat = self.cat_enc.inverse_transform(normal_x_df_cat).toarray()
        x_ord_cont = self.scaler.inverse_transform(normal_x_df_ord_cont).toarray()
        x = np.concatenate((x_bin, x_cat, x_ord_cont), axis=1)
        return x

    """
    MACE Methodology methods / classes (based on Model-Agnostic Counterfactual Explanations (MACE) authors implementation: See https://github.com/amirhk/mace)
    """

    def getAttributeNames(self, allowed_node_types, long_or_kurz = 'kurz'):
        names = []
        # We must loop through all attributes and check attr_name
        for attr_name in self.attributes_long.keys():
            attr_obj = self.attributes_long[attr_name]
            if attr_obj.node_type not in allowed_node_types:
                continue
            if long_or_kurz == 'long':
                names.append(attr_obj.attr_name_long)
            elif long_or_kurz == 'kurz':
                names.append(attr_obj.attr_name_kurz)
            else:
                raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getInputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'input'}, long_or_kurz)
    
    def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'input', 'output'}, long_or_kurz)

def load_dataset(data_str, train_fraction, seed, step):
    """
    Function to load all datasets according to data_str and train_fraction
    """
    if data_str == 'adult':
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        ordinal = ['EducationLevel','AgeGroup']
        continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek']
        input_cols = binary + categorical + ordinal + continuous
        label = ['label']
        df = pd.read_csv(dataset_dir+'adult/preprocessed_adult.csv', index_col=0)
        
        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'NativeCountry':
                attr_type = 'binary'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'Race':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'WorkClass':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'EducationNumber':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True
            elif col_name == 'EducationLevel':
                attr_type = 'ordinal'
                actionability = 'any'
                mutability = True
            elif col_name == 'MaritalStatus':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Occupation':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Relationship':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'CapitalGain':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'CapitalLoss':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'HoursPerWeek':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True

            attributes_df[col_name] = DatasetAttribute(
                attr_name_long = col_name,
                attr_name_kurz = f'x{col_idx}',
                attr_type = attr_type,
                node_type = 'input',
                actionability = actionability,
                mutability = mutability,
                parent_name_long = -1,
                parent_name_kurz = -1,
                lower_bound = df[col_name].min(),
                upper_bound = df[col_name].max())

    elif data_str == 'ionosphere':
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
        df = pd.read_csv(dataset_dir+'synthetic_disease/processed_synthetic_disease.csv',index_col=0)
    
    data_obj = Dataset(data_str, seed, train_fraction, label, df,
                   binary, categorical, ordinal, continuous, step, attributes_df)
    return data_obj

"""
MACE Methodology methods / classes (based on Model-Agnostic Counterfactual Explanations (MACE) authors implementation: See https://github.com/amirhk/mace)
"""

class DatasetAttribute():

    def __init__(self, attr_name_long, attr_name_kurz, attr_type, node_type,
                 actionability, mutability, parent_name_long, parent_name_kurz, lower_bound,
                 upper_bound):

        if attr_type in {'sub-categorical', 'sub-ordinal'}:
            assert parent_name_long != -1, 'Parent ID set for non-hot attribute.'
            assert parent_name_kurz != -1, 'Parent ID set for non-hot attribute.'
            if attr_type == 'sub-categorical':
                assert lower_bound == 0
                assert upper_bound == 1
            if attr_type == 'sub-ordinal':
                # the first elem in thermometer is always on, but the rest may be on or off
                assert lower_bound == 0 or lower_bound == 1
                assert upper_bound == 1
        else:
            assert parent_name_long == -1, 'Parent ID set for non-hot attribute.'
            assert parent_name_kurz == -1, 'Parent ID set for non-hot attribute.'

        if attr_type in {'categorical', 'ordinal'}:
            assert lower_bound == 1 # setOneHotValue & setThermoValue assume this in their logic

        if attr_type in {'binary', 'categorical', 'sub-categorical'}: # not 'ordinal' or 'sub-ordinal'
            # IMPORTANT: surprisingly, it is OK if all sub-ordinal variables share actionability
            #            think about it, if each sub- variable is same-or-increase, along with
            #            the constraints that x0_ord_1 >= x0_ord_2, all variables can only stay
            #            the same or increase. It works :)
            assert actionability in {'none', 'any'}, f"{attr_type}'s actionability can only be in {'none', 'any'}, not `{actionability}`."

        if node_type != 'input':
            assert actionability == 'none', f'{node_type} attribute is not actionable.'
            assert mutability == False, f'{node_type} attribute is not mutable.'

        # We have introduced 3 types of variables: (actionable and mutable, non-actionable but mutable, immutable and non-actionable)
        if actionability != 'none':
            assert mutability == True
        # TODO: above/below seem contradictory... (2020.04.14)
        if mutability == False:
            assert actionability == 'none'

        if parent_name_long == -1 or parent_name_kurz == -1:
            assert parent_name_long == parent_name_kurz == -1

        self.attr_name_long = attr_name_long
        self.attr_name_kurz = attr_name_kurz
        self.attr_type = attr_type
        self.node_type = node_type
        self.actionability = actionability
        self.mutability = mutability
        self.parent_name_long = parent_name_long
        self.parent_name_kurz = parent_name_kurz
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
