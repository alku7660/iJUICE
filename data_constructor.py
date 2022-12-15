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
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.features = binary + categorical + ordinal + continuous
        self.df = df[self.features + self.label_name]
        self.step = step
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.df, self.df[self.label_name], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target = self.balance_train_data()
        self.bin_enc, self.cat_enc, self.bin_cat_enc, self.scaler = self.encoder_scaler_fit()
        self.bin_enc_cols, self.cat_enc_cols, self.bin_cat_enc_cols = self.encoder_scaler_cols()
        self.processed_features = list(self.bin_enc_cols) + list(self.cat_enc_cols) + self.ordinal + self.continuous
        self.transformed_train_df = self.transform_data(self.train_df)
        self.transformed_train_np = self.transformed_train_df.to_numpy()
        self.transformed_test_df = self.transform_data(self.test_df)
        self.transformed_test_np = self.transformed_test_df.to_numpy()
        self.train_target, self.test_target = self.change_targets_to_numpy()
        self.undesired_class = self.undesired_class_data()
        self.feat_type, self.feat_type_mace = self.define_feat_type()
        self.feat_mutable = self.define_feat_mutability()
        self.immutables = self.get_immutables()
        self.feat_directionality, self.feat_directionality_mace = self.define_feat_directionality()
        self.feat_step = self.define_feat_step()
        self.feat_cat = self.define_feat_cat()
        self.idx_cat_cols_dict = self.idx_cat_columns()
        self.feat_dist, self.processed_feat_dist = self.feature_distribution()
        
        """
        MACE attributes
        """
        self.is_one_hot = True # Set to True always since we always one-hot encode
        self.data_frame_long = self.df
        self.attributes_long = self.define_attributes()
        attributes_kurz = dict((self.attributes_long[key].attr_name_kurz, value) for (key, value) in self.attributes_long.items())
        transformed_train_df_target = copy.deepcopy(self.transformed_train_df)
        transformed_test_df_target = copy.deepcopy(self.transformed_test_df)
        transformed_train_df_target[self.label_name[0]] = self.train_target
        transformed_test_df_target[self.label_name[0]] = self.test_target
        all_data = pd.concat((transformed_train_df_target, transformed_test_df_target), axis=0)
        data_frame_kurz = copy.deepcopy(all_data)
        data_frame_kurz.columns = self.getAllAttributeNames('kurz')
        self.data_frame_kurz = data_frame_kurz # i.e., data_frame is indexed by attr_name_kurz
        self.attributes_kurz = attributes_kurz # i.e., attributes is indexed by attr_name_kurz

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
        bin_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        cat_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        bin_cat_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        scaler = MinMaxScaler(clip=True)
        bin_enc.fit(self.train_df[self.binary])
        cat_enc.fit(self.train_df[self.categorical])
        bin_cat_enc.fit(self.train_df[self.binary + self.categorical])
        scaler.fit(self.train_df[self.ordinal + self.continuous])
        return bin_enc, cat_enc, bin_cat_enc, scaler

    def encoder_scaler_cols(self):
        """
        Method that extracts the encoded columns from the encoders
        """
        return self.bin_enc.get_feature_names_out(self.binary), self.cat_enc.get_feature_names_out(self.categorical), self.bin_cat_enc.get_feature_names_out(self.binary + self.categorical)

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
        if self.name in ['compass','credit','german','heart','synthetic_disease','diabetes']:
            undesired_class = 1
        elif self.name in ['ionosphere','adult','kdd_census','dutch','bank','synthetic_athlete','student','oulad','law']:
            undesired_class = 0
        return undesired_class
    
    def undesired_test(self, model):
        """
        Method to obtain the test subset with predicted undesired class
        """
        self.undesired_transformed_test_df = self.transformed_test_df.loc[model.model.predict(self.transformed_test_np) == self.undesired_class]
        self.undesired_transformed_test_np = self.undesired_transformed_test_df.to_numpy()

    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        """
        feat_type = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_type_mace = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Sex' in i or 'Native' in i or 'Race' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relationship' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Industry' in i or 'Occupation' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Age' in i or 'WageHour' in i or 'Capital' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Sex' in i or 'Race' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Industry' in i or 'Occupation' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i or 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i or 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type.loc[i] = 'bin'
                elif 'EducationLevel' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'Age' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Sex' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Age' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Default' in i or 'Housing' in i or 'Loan' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type.loc[i] = 'bin'
                    feat_type_mace.loc[i] = 'binary'
                elif 'Total' in i or 'Age' in i or 'Education' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type.loc[i] = 'bin'
                    feat_type_mace.loc[i] = 'binary'
                elif 'Priors' in i or 'Age' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'DiabetesMed' in i or 'Race' in i or 'Sex' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type.loc[i] = 'bin'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency' in i:
                    feat_type.loc[i] = 'cont'
                if 'DiabetesMed' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Race' in i or 'Sex' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
                elif 'TimeInHospital' in i:
                    feat_type_mace.loc[i] = 'numeric-real'
                elif 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency' in i:
                    feat_type_mace.loc[i] = 'numeric-int'
        elif self.name == 'student':
            for i in feat_list:
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i or 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type.loc[i] = 'bin'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i:
                    feat_type.loc[i] = 'cont'
                if 'School' in i or 'Sex' in i or 'Age' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i or 'NumEmergency' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
                elif 'ClassFailures' in i or 'GoOut' in i:
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'TravelTime' in i :
                    feat_type_mace.loc[i] = 'numeric-real'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i or 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type.loc[i] = 'bin'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type.loc[i] = 'cont'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                if 'Sex' in i or 'Disability' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
                elif 'NumPrevAttempts' in i:
                    feat_type_mace.loc[i] = 'numeric-int'
                elif 'StudiedCredits' in i:
                    feat_type_mace.loc[i] = 'numeric-real'
        elif self.name == 'law':
            for i in feat_list:
                if 'FamilyIncome' in i or 'Tier' in i or 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'WorkFullTime' in i or 'Sex' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'FamilyIncome' in i or 'Tier' in i or 'Race' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
                feat_type_mace.loc[i] = 'numeric-real'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                    feat_type_mace.loc[i] = 'binary'
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i:
                    feat_type.loc[i] = 'cat'
                    feat_type_mace.loc[i] = 'sub-categorical'
                elif i in ['Age','SleepHours']:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i or 'Training' in i or 'Sport' in i or 'Diet' in i:
                    feat_type.loc[i] = 'bin'
                elif i in ['Age','SleepHours']:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Sex' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Training' in i or 'Sport' in i or 'Diet' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Smokes' in i or 'Diet' in i or 'Stress' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Weight' in i:
                    feat_type.loc[i] = 'ord'
                    feat_type_mace.loc[i] = 'numeric-int'
                elif i in ['Age','ExerciseMinutes','SleepHours']:
                    feat_type.loc[i] = 'cont'
                    feat_type_mace.loc[i] = 'numeric-real'
                if 'Smokes' in i:
                    feat_type_mace.loc[i] = 'binary'
                elif 'Diet' in i or 'Stress' in i:
                    feat_type_mace.loc[i] = 'sub-categorical'
        return feat_type, feat_type_mace

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
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i or 'Country' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'bank':
            for i in feat_list:
                if 'AgeGroup' in i or 'Marital' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'credit':
            for i in feat_list:
                if 'isMale' in i or 'isMarried' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'compass':
            for i in feat_list:
                if 'Race' in i or 'Sex' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'student':
            for i in feat_list:
                if 'Sex' in i or 'AgeGroup':
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'law':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
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
                if i in ['Age', 'Sex']:
                    feat_mutable[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_mutable[i] = 1
        elif self.name == 'synthetic_disease':
            for i in feat_list:
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
        feat_directionality_mace = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_directionality.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i or 'Native' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'Education' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase'
                else:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' or 'Age' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_directionality[i] = 0
                else:
                    feat_directionality[i] = 'any'
                if 'Age' in i:
                    feat_directionality_mace[i] = 'same-or-increase'
                elif 'Sex' in i:
                    feat_directionality_mace[i] = 'none'
                else:
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i or 'Country' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Age' in i or 'Marital' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'Contact' in i or 'Month' in i or 'Poutcome' or 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'Total' in i or 'History' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
                if 'Age' in i or 'Education' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase'
                elif 'Charge' in i or 'Priors' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase'
                else:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any'
        elif self.name == 'student':
            for i in feat_list:
                if 'Sex' in i or 'Age' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_directionality[i] = 'pos'
                    feat_directionality_mace[i] = 'same-or-increase' 
                else:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any' 
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                else:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any' 
        elif self.name == 'law':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                else:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any' 
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_directionality[i] = 'any'
                feat_directionality_mace[i] = 'any'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_directionality[i] = 0
                    feat_directionality_mace[i] = 'none'
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_directionality[i] = 'any'
                    feat_directionality_mace[i] = 'any' 
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i:
                    feat_directionality[i] = 'pos'
                elif 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i or 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_directionality[i] = 'any'
        return feat_directionality, feat_directionality_mace

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
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'EducationLevel' in i or 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'Race' in i:
                    feat_cat.loc[i] = 'non'
                elif 'Age' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'WorkClass' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Marital' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Occupation' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Relation' in i:
                    feat_cat.loc[i] = 'cat_4'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Industry' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'Occupation' in i:
                    feat_cat.loc[i] = 'cat_1'    
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'german':
            for i in feat_list:
                if 'PurposeOfLoan' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'InstallmentRate' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Housing' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'HouseholdPosition' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'HouseholdSize' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Country' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'EconomicStatus' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'CurEcoActivity' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'MaritalStatus' in i:
                    feat_cat.loc[i] = 'cat_5'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Job' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'MaritalStatus' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Education' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Contact' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Month' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'Poutcome' in i:
                    feat_cat.loc[i] = 'cat_5'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'credit':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'compass':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Race' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'A1CResult' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Metformin' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Chlorpropamide' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Glipizide' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'Rosiglitazone' in i:
                    feat_cat.loc[i] = 'cat_5'
                elif 'Acarbose' in i:
                    feat_cat.loc[i] = 'cat_6'
                elif 'Miglitol' in i:
                    feat_cat.loc[i] = 'cat_7'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'student':
            for i in feat_list:
                if 'MotherJob' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'FatherJob' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'SchoolReason' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Region' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'CodeModule' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'CodePresentation' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'HighestEducation' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'IMDBand' in i:
                    feat_cat.loc[i] = 'cat_4'
                else:
                    feat_cat.loc[i] = 'non'    
        elif self.name == 'law':
            for i in feat_list:
                if 'FamilyIncome' in i:   
                    feat_cat.loc[i] = 'cat_0'
                elif 'Tier' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Race' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'ionosphere':
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

    def inverse(self, normal_x, mace=False):
        """
        Method that transforms an instance back into the original space
        """
        if mace:
            x_df = copy.deepcopy(normal_x)
            for col in self.categorical:
                mace_cat_cols = [i for i in x_df.columns if '_cat_' in i and col in i]
                for mace_col in mace_cat_cols:
                    if x_df[mace_col].values[0] == 1:
                        col_name_value = mace_col.split('_cat_')
                        col_name, col_value = col_name_value[0], int(col_name_value[1]) + 1
                        break
                x_df[col_name] = col_value
                x_df.drop(mace_cat_cols, axis=1, inplace=True)
            for col in self.ordinal:
                mace_ord_cols = [i for i in x_df.columns if '_ord_' in i and col in i]
                current_col_with_1 = 0
                for ord_col in mace_ord_cols:
                    if x_df[ord_col].values[0] == 1:
                        current_col_with_1 = ord_col
                    elif x_df[ord_col].values[0] == 0:
                        col_name_value = current_col_with_1.split('_ord_')
                        col_name, col_value = col_name_value[0], int(col_name_value[1]) + 1
                        break
                x_df[col_name] = col_value
                x_df.drop(mace_ord_cols, axis=1, inplace=True)
            x = x_df[self.features]                  
        else:
            normal_x_df = pd.DataFrame(data=normal_x.reshape(1, -1), columns=self.processed_features)
            normal_x_df_bin, normal_x_df_cat, normal_x_df_ord_cont = normal_x_df[self.bin_enc_cols], normal_x_df[self.cat_enc_cols], normal_x_df[self.ordinal+self.continuous]
            try:
                x_bin = self.bin_enc.inverse_transform(normal_x_df_bin)
            except:
                x_bin = np.array([[]])
            try:
                x_cat = self.cat_enc.inverse_transform(normal_x_df_cat)
            except:
                x_cat = np.array([[]])
            try:
                x_ord_cont = self.scaler.inverse_transform(normal_x_df_ord_cont)
            except:
                x_ord_cont = np.array([[]])
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
    
    def getOutputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'output'}, long_or_kurz)

    def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'input', 'output'}, long_or_kurz)
    
    def getAllAttributeNames(self, long_or_kurz = 'kurz'):
        return self.getAttributeNames({'meta', 'input', 'output'}, long_or_kurz)
    
    def getDictOfSiblings(self, long_or_kurz = 'kurz'):
        if long_or_kurz == 'long':
            dict_of_siblings_long = {}
            dict_of_siblings_long['cat'] = {}
            dict_of_siblings_long['ord'] = {}
            for attr_name_long in self.getInputAttributeNames('long'):
                attr_obj = self.attributes_long[attr_name_long]
                if attr_obj.attr_type == 'sub-categorical':
                    if attr_obj.parent_name_long not in dict_of_siblings_long['cat'].keys():
                        dict_of_siblings_long['cat'][attr_obj.parent_name_long] = [] # initiate key-value pair
                    dict_of_siblings_long['cat'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
                elif attr_obj.attr_type == 'sub-ordinal':
                    if attr_obj.parent_name_long not in dict_of_siblings_long['ord'].keys():
                        dict_of_siblings_long['ord'][attr_obj.parent_name_long] = [] # initiate key-value pair
                    dict_of_siblings_long['ord'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
            # sort sub-arrays
            for key in dict_of_siblings_long['cat'].keys():
                dict_of_siblings_long['cat'][key] = sorted(dict_of_siblings_long['cat'][key], key = lambda x : int(float(x.split('_')[-1])))
            for key in dict_of_siblings_long['ord'].keys():
                dict_of_siblings_long['ord'][key] = sorted(dict_of_siblings_long['ord'][key], key = lambda x : int(float(x.split('_')[-1])))
            return dict_of_siblings_long
        elif long_or_kurz == 'kurz':
            dict_of_siblings_kurz = {}
            dict_of_siblings_kurz['cat'] = {}
            dict_of_siblings_kurz['ord'] = {}
            for attr_name_kurz in self.getInputAttributeNames('kurz'):
                attr_obj = self.attributes_kurz[attr_name_kurz]
                if attr_obj.attr_type == 'sub-categorical':
                    if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['cat'].keys():
                        dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
                    dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
                elif attr_obj.attr_type == 'sub-ordinal':
                    if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['ord'].keys():
                        dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
                    dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
            # sort sub-arrays
            for key in dict_of_siblings_kurz['cat'].keys():
                dict_of_siblings_kurz['cat'][key] = sorted(dict_of_siblings_kurz['cat'][key], key = lambda x : int(float(x.split('_')[-1])))
            for key in dict_of_siblings_kurz['ord'].keys():
                dict_of_siblings_kurz['ord'][key] = sorted(dict_of_siblings_kurz['ord'][key], key = lambda x : int(float(x.split('_')[-1])))
            return dict_of_siblings_kurz
        else:
            raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

    def getSiblingsFor(self, attr_name_long_or_kurz):
    # If attr_name_long is given, we will return siblings_long (the same length)
    # but not siblings_kurz. Same for the opposite direction.
        # assert \
        #     'cat' in attr_name_long_or_kurz or 'ord' in attr_name_long_or_kurz, \
        #     'attr_name must include either `cat` or `ord`.'
        if attr_name_long_or_kurz in self.getInputOutputAttributeNames('long'):
            attr_name_long = attr_name_long_or_kurz
            dict_of_siblings_long = self.getDictOfSiblings('long')
            for parent_name_long in dict_of_siblings_long['cat']:
                siblings_long = dict_of_siblings_long['cat'][parent_name_long]
                if attr_name_long_or_kurz in siblings_long:
                    return siblings_long
            for parent_name_long in dict_of_siblings_long['ord']:
                siblings_long = dict_of_siblings_long['ord'][parent_name_long]
                if attr_name_long_or_kurz in siblings_long:
                    return siblings_long
        elif attr_name_long_or_kurz in self.getInputOutputAttributeNames('kurz'):
            attr_name_kurz = attr_name_long_or_kurz
            dict_of_siblings_kurz = self.getDictOfSiblings('kurz')
            for parent_name_kurz in dict_of_siblings_kurz['cat']:
                siblings_kurz = dict_of_siblings_kurz['cat'][parent_name_kurz]
                if attr_name_long_or_kurz in siblings_kurz:
                    return siblings_kurz
            for parent_name_kurz in dict_of_siblings_kurz['ord']:
                siblings_kurz = dict_of_siblings_kurz['ord'][parent_name_kurz]
                if attr_name_long_or_kurz in siblings_kurz:
                    return siblings_kurz
        else:
            raise Exception(f'{attr_name_long_or_kurz} not recognized as a valid `attr_name_long_or_kurz`.')

    def getMutableAttributeNames(self, long_or_kurz = 'kurz'):
        names = []
        # We must loop through all attributes and check mutability
        for attr_name_long in self.getInputAttributeNames('long'):
            attr_obj = self.attributes_long[attr_name_long]
            if attr_obj.node_type == 'input' and attr_obj.mutability != False:
                if long_or_kurz == 'long':
                    names.append(attr_obj.attr_name_long)
                elif long_or_kurz == 'kurz':
                    names.append(attr_obj.attr_name_kurz)
                else:
                    raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
        return np.array(names)

    def getOneHotAttributesNames(self, long_or_kurz = 'kurz'):
        tmp = self.getDictOfSiblings(long_or_kurz)
        names = []
        for key1 in tmp.keys():
            for key2 in tmp[key1].keys():
                names.extend(tmp[key1][key2])
        return np.array(names)

    def getNonHotAttributesNames(self, long_or_kurz = 'kurz'):
        a = self.getInputAttributeNames(long_or_kurz)
        b = self.getOneHotAttributesNames(long_or_kurz)
        return np.setdiff1d(a,b)

    def define_attributes(self):
        """
        Method that defines the attributes based on the MACE methodology and the loading for the rest of the methods
        """
        attributes = {}        
        old_attributes = list(self.train_df.columns)
        new_attributes = list(self.transformed_train_df.columns)

        for old_col_idx, old_col in enumerate(old_attributes):

            any_old_col = [col for col in self.feat_type_mace.keys() if old_col in col][0]
            old_col_type = self.feat_type_mace[any_old_col]
        
            if old_col_type in ['binary', 'numeric-int', 'numeric-real']:
                parent_name_long = -1
                parent_name_kurz = -1
            else:
                parent_name_long = -1
                parent_name_kurz = -1
                old_col_type = 'categorical'

            old_col_actionability = self.feat_directionality_mace[any_old_col]
            old_col_mutability = True if self.feat_directionality_mace[any_old_col] != 'none' else False
            attributes[old_col] = DatasetAttribute(
                attr_name_long = old_col,
                attr_name_kurz = f'x{old_col_idx}',
                attr_type = old_col_type,
                node_type = 'input',
                actionability = old_col_actionability,
                mutability = old_col_mutability,
                parent_name_long = parent_name_long,
                parent_name_kurz = parent_name_kurz,
                lower_bound = self.train_df[old_col].min(),
                upper_bound = self.train_df[old_col].max())

        for new_col in new_attributes:
            
            old_col = [col for col in attributes.keys() if col in new_col][0]
            new_attr_type = self.feat_type_mace[new_col]
            old_col_name_long = old_col
            old_attr_name_long = attributes[old_col_name_long].attr_name_long
            old_attr_name_kurz = attributes[old_col_name_long].attr_name_kurz
            old_attr_type = attributes[old_col_name_long].attr_type
            old_node_type = attributes[old_col_name_long].node_type
            old_actionability = attributes[old_col_name_long].actionability
            old_mutability = attributes[old_col_name_long].mutability
            old_parent_name_long = attributes[old_col_name_long].parent_name_long
            old_parent_name_kurz = attributes[old_col_name_long].parent_name_kurz
            if 'categorical' in new_attr_type or 'ordinal' in new_attr_type:
                if 'categorical' in new_attr_type:
                    feat_str = 'cat'
                elif 'ordinal' in new_attr_type:
                    feat_str = 'ord'
                col_unique_val = int(float(new_col.split('_')[-1]))
                new_attr_type = new_attr_type
                new_node_type = old_node_type
                new_actionability = old_actionability
                new_mutability = old_mutability
                new_parent_name_long = old_attr_name_long
                new_parent_name_kurz = old_attr_name_kurz
                attributes[new_col] = DatasetAttribute(
                    attr_name_long = new_col,
                    attr_name_kurz = f'{old_attr_name_kurz}_{feat_str}_{col_unique_val}',
                    attr_type = new_attr_type,
                    node_type = new_node_type,
                    actionability = new_actionability,
                    mutability = new_mutability,
                    parent_name_long = new_parent_name_long,
                    parent_name_kurz = new_parent_name_kurz,
                    lower_bound = self.transformed_train_df[new_col].min(),
                    upper_bound = self.transformed_train_df[new_col].max())
            elif 'numeric' in new_attr_type:
                del attributes[old_col]
                attributes[new_col] = DatasetAttribute(
                    attr_name_long = old_attr_name_long,
                    attr_name_kurz = old_attr_name_kurz,
                    attr_type = new_attr_type,
                    node_type = old_node_type,
                    actionability = old_actionability,
                    mutability = old_mutability,
                    parent_name_long = old_parent_name_long,
                    parent_name_kurz = old_parent_name_kurz,
                    lower_bound = self.transformed_train_df[new_col].min(),
                    upper_bound = self.transformed_train_df[new_col].max())

        for old_col in old_attributes:
            any_old_col = [col for col in self.feat_type_mace.keys() if old_col in col][0]
            old_col_type = self.feat_type_mace[any_old_col]
            if old_col_type in ['sub-categorical','sub-ordinal']:
                del attributes[old_col]

        col_name = self.label_name[0]
        attributes[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = min(self.train_target), upper_bound = max(self.train_target))

        return attributes
                
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
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'EducationLevel':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
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
                
    elif data_str == 'kdd_census':
        binary = ['Sex','Race']
        categorical = ['Industry','Occupation']
        ordinal = []
        continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'kdd_census/preprocessed_kdd_census.csv', index_col=0)
    
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
            elif col_name == 'Race':
                attr_type = 'binary'
                actionability = 'none' # 'none'
                mutability = False
            elif col_name == 'Industry':
                attr_type = 'categorical'
                actionability = 'any' # 'none'
                mutability = True
            elif col_name == 'Occupation':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'WageHour':
                attr_type = 'numeric-real'
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
            elif col_name == 'Dividends':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'WorkWeeksYear':
                attr_type = 'numeric-real'
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
    
    elif data_str == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        ordinal = []
        continuous = ['Age','Credit','LoanDuration']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'german/preprocessed_german.csv', index_col=0)
    
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
                actionability = 'none'
                mutability = False
            elif col_name == 'Single':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Unemployed':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'PurposeOfLoan':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'InstallmentRate':
                attr_type = 'categorical'
                actionability = 'Any'
                mutability = True
            elif col_name == 'Housing':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'Credit':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'LoanDuration':
                attr_type = 'numeric-real'
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

    elif data_str == 'dutch':
        binary = ['Sex']
        categorical = ['HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
        ordinal = ['EducationLevel']
        continuous = ['Age']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Occupation']
        df = pd.read_csv(dataset_dir+'dutch/preprocessed_dutch.csv', index_col=0)
    
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
                actionability = 'none'
                mutability = False
            elif col_name == 'HouseholdPosition':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'HouseholdSize':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Country':
                attr_type = 'categorical'
                actionability = 'none'
                mutability = False
            elif col_name == 'EconomicStatus':
                attr_type = 'categorical'
                actionability = 'Any'
                mutability = True
            elif col_name == 'CurEcoActivity':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'MaritalStatus':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'EducationLevel':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'same-or-increase'
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
    
    elif data_str == 'bank':
        binary = ['Default','Housing','Loan']
        categorical = ['Job','MaritalStatus','Education','Contact','Month','Poutcome']
        ordinal = ['AgeGroup']
        continuous = ['Balance','Day','Duration','Campaign','Pdays','Previous']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Subscribed']
        df = pd.read_csv(dataset_dir+'bank/preprocessed_bank.csv', index_col=0)

        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Default':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Housing':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Loan':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Job':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'MaritalStatus':
                attr_type = 'categorical'
                actionability = 'none'
                mutability = False
            elif col_name == 'Education':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Contact':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Month':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Poutcome':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'none'
                mutability = False
            elif col_name == 'Balance':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Day':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Duration':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Campaign':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Pdays':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Previous':
                attr_type = 'numeric-real'
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

    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        ordinal = ['TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount']
        input_cols = binary + categorical + ordinal + continuous
        label = ['NoDefaultNextMonth (label)']
        df = pd.read_csv(dataset_dir+'/credit/preprocessed_credit.csv')

        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'isMale':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'isMarried':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'HasHistoryOfOverduePayments':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'TotalOverdueCounts':
                attr_type = 'ordinal'
                actionability = 'any'
                mutability = True
            elif col_name == 'TotalMonthsOverdue':
                attr_type = 'ordinal'
                actionability = 'any'
                mutability = True
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'EducationLevel':
                attr_type = 'ordinal'
                actionability = 'any'
                mutability = True
            elif col_name == 'MaxBillAmountOverLast6Months':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MaxPaymentAmountOverLast6Months':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MonthsWithZeroBalanceOverLast6Months':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MonthsWithLowSpendingOverLast6Months':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MonthsWithHighSpendingOverLast6Months':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MostRecentBillAmount':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'MostRecentPaymentAmount':
                attr_type = 'numeric-real'
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

    elif data_str == 'compass':
        # Based on the MACE algorithm Datasets preprocessing (please, see: https://github.com/amirhk/mace)
        df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        ordinal = ['PriorsCount','AgeGroup']
        continuous = []
        input_cols = binary + categorical + ordinal + continuous
        label = ['TwoYearRecid (label)']
        df = pd.read_csv(dataset_dir+'/compass/preprocessed_compass.csv')
    
        """
        MACE variables / attributes
        """
        # attributes_df = {}
        # col_name = label[0]
        # attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
        #                                            mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        # for col_idx, col_name in enumerate(input_cols):

        #     if col_name == 'Race':
        #         attr_type = 'binary'
        #         actionability = 'none'
        #         mutability = False
        #     elif col_name == 'Sex':
        #         attr_type = 'binary'
        #         actionability = 'none'
        #         mutability = False
        #     elif col_name == 'ChargeDegree':
        #         attr_type = 'binary'
        #         actionability = 'any'
        #         mutability = True
        #     elif col_name == 'PriorsCount':
        #         attr_type = 'ordinal'
        #         actionability = 'any'
        #         mutability = True
        #     elif col_name == 'AgeGroup':
        #         attr_type = 'ordinal'
        #         actionability = 'same-or-increase'
        #         mutability = True
            
        #     attributes_df[col_name] = DatasetAttribute(
        #         attr_name_long = col_name,
        #         attr_name_kurz = f'x{col_idx}',
        #         attr_type = attr_type,
        #         node_type = 'input',
        #         actionability = actionability,
        #         mutability = mutability,
        #         parent_name_long = -1,
        #         parent_name_kurz = -1,
        #         lower_bound = df[col_name].min(),
        #         upper_bound = df[col_name].max())
    
    elif data_str == 'diabetes':
        binary = ['DiabetesMed']
        categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol']
        ordinal = ['AgeGroup']
        continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'/diabetes/preprocessed_diabetes.csv')
    
        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'DiabetesMed':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Race':
                attr_type = 'categorical'
                actionability = 'none'
                mutability = False
            elif col_name == 'Sex':
                attr_type = 'categorical'
                actionability = 'none'
                mutability = False
            elif col_name == 'A1CResult':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Metformin':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Chlorpropamide':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Glipizide':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Rosiglitazone':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Acarbose':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Miglitol':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'TimeInHospital':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'NumProcedures':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True
            elif col_name == 'NumMedications':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True
            elif col_name == 'NumEmergency':
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

    elif data_str == 'student':
        binary = ['School','Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
        categorical = ['MotherJob','FatherJob','SchoolReason']
        ordinal = ['MotherEducation','FatherEducation']
        continuous = ['TravelTime','ClassFailures','GoOut']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Grade']
        df = pd.read_csv(dataset_dir+'/student/preprocessed_student.csv')
    
        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'School':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'AgeGroup':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'Address':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'FamilySize':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'ParentStatus':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'SchoolSupport':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'FamilySupport':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'ExtraPaid':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'ExtraActivities':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Nursery':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'HigherEdu':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Internet':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Romantic':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'NumEmergency':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'MotherJob':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'FatherJob':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'SchoolReason':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'MotherEducation':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'FatherEducation':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'TravelTime':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'ClassFailures':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True
            elif col_name == 'GoOut':
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

    elif data_str == 'oulad':
        binary = ['Sex','Disability']
        categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
        ordinal = ['AgeGroup']
        continuous = ['NumPrevAttempts','StudiedCredits']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Grade']
        df = pd.read_csv(dataset_dir+'/oulad/preprocessed_oulad.csv')

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
                actionability = 'none'
                mutability = False
            elif col_name == 'Disability':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'Region':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'CodeModule':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'CodePresentation':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'HighestEducation':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'IMDBand':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'AgeGroup':
                attr_type = 'ordinal'
                actionability = 'same-or-increase'
                mutability = True
            elif col_name == 'NumPrevAttempts':
                attr_type = 'numeric-int'
                actionability = 'any'
                mutability = True
            elif col_name == 'StudiedCredits':
                attr_type = 'numeric-real'
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

    elif data_str == 'law':
        binary = ['WorkFullTime','Sex']
        categorical = ['FamilyIncome','Tier','Race']
        ordinal = []
        continuous = ['Decile1stYear','Decile3rdYear','LSAT','UndergradGPA','FirstYearGPA','CumulativeGPA']
        input_cols = binary + categorical + ordinal + continuous
        label = ['BarExam']
        df = pd.read_csv(dataset_dir+'/law/preprocessed_law.csv')

        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'WorkFullTime':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Sex':
                attr_type = 'binary'
                actionability = 'none'
                mutability = False
            elif col_name == 'FamilyIncome':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Tier':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Race':
                attr_type = 'categorical'
                actionability = 'none'
                mutability = False
            elif col_name == 'Decile1stYear':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'Decile3rdYear':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'LSAT':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'UndergradGPA':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'FirstYearGPA':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'CumulativeGPA':
                attr_type = 'numeric-real'
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
        input_cols = binary + categorical + ordinal + continuous
        label = ['label']
        df = pd.read_csv(dataset_dir+'/ionosphere/processed_ionosphere.csv',index_col=0)
    
        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == '0':
                attr_type = 'numeric-real'
                actionability = 'none'
                mutability = False
            elif col_name == '2':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '4':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '5':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '6':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '7':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '26':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == '30':
                attr_type = 'numeric-real'
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
    
    elif data_str == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        ordinal = []
        continuous = ['Age','SleepHours']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'synthetic_athlete/processed_synthetic_athlete.csv',index_col=0)
    
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
                actionability = 'none'
                mutability = False
            elif col_name == 'Diet':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Sport':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'TrainingTime':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'none'
                mutability = False
            elif col_name == 'SleepHours':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True

            attributes_df[col_name] = DatasetAttribute(
                attr_name_long = col_name,
                attr_name_kurz = col_name,
                attr_type = attr_type,
                node_type = 'input',
                actionability = actionability,
                mutability = mutability,
                parent_name_long = -1,
                parent_name_kurz = -1,
                lower_bound = df[col_name].min(),
                upper_bound = df[col_name].max())
    
    elif data_str == 'synthetic_disease':
        binary = ['Smokes']
        categorical = ['Diet','Stress']
        ordinal = ['Weight']
        continuous = ['Age','ExerciseMinutes','SleepHours']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'synthetic_disease/processed_synthetic_disease.csv',index_col=0)
    
        """
        MACE variables / attributes
        """
        attributes_df = {}
        col_name = label[0]
        attributes_df[col_name] = DatasetAttribute(attr_name_long = col_name, attr_name_kurz = 'y', attr_type = 'binary', node_type = 'output', actionability = 'none',
                                                   mutability = False, parent_name_long = -1, parent_name_kurz = -1, lower_bound = df[col_name].min(), upper_bound = df[col_name].max())
        for col_idx, col_name in enumerate(input_cols):

            if col_name == 'Smokes':
                attr_type = 'binary'
                actionability = 'any'
                mutability = True
            elif col_name == 'Diet':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Stress':
                attr_type = 'categorical'
                actionability = 'any'
                mutability = True
            elif col_name == 'Weight':
                attr_type = 'ordinal'
                actionability = 'any'
                mutability = True
            elif col_name == 'Age':
                attr_type = 'numeric-real'
                actionability = 'same-or-increase'
                mutability = False
            elif col_name == 'ExerciseMinutes':
                attr_type = 'numeric-real'
                actionability = 'any'
                mutability = True
            elif col_name == 'SleepHours':
                attr_type = 'numeric-real'
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

    data_obj = Dataset(data_str, seed, train_fraction, label, df,
                   binary, categorical, ordinal, continuous, step)
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
