"""
C-CHVAE
Based on:
    (1) original authors implementation: Please see https://github.com/MartinPawel/c-chvae
    (2) CARLA framework: Please see https://github.com/carla-recourse/CARLA
"""

"""
Imports
"""
from abc import ABC, abstractmethod
from address import results_obj
from address import dataset_dir
from typing import Union
import numpy as np
import pandas as pd

class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user.
    """
    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.
        """
        pass

    @property
    @abstractmethod
    def df(self):
        """
        The full Dataframe.
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Data transformation, for example normalization of continuous features and encoding of categorical features.
        """
        pass

    @abstractmethod
    def inverse_transform(self, df):
        """
        Inverts transform operation.
        """
        pass

class MLModel(ABC):
    """
    Abstract class to implement custom black-box-model for a given dataset with encoding and scaling processing.
    """

    def __init__(self, data: Data):
        self._data: Data = data

    @property
    def data(self) -> Data:
        """
        Contains the data.api.Data dataset.
        """
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of features as list.
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the classifier.
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Contains the raw ML model built on its framework
        """
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].
        """
        pass

    @abstractmethod
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        Two-dimensional probability prediction of ml model.
        """
        pass

    def get_ordered_features(self, x):
        """
        Restores the correct input feature order for the ML model, this also drops the target column.
        """
        if isinstance(x, pd.DataFrame):
            return order_data(self.feature_input_order, x)
        else:
            warnings.warn(
                f"cannot re-order features for non dataframe input: {type(x)}"
            )
            return x

class CCHVAE:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = cchvae_method(counterfactual)

def order_data(feature_order: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Restores the correct input feature order for the ML model

    Only works for encoded data

    Parameters
    ----------
    feature_order : list
        List of input feature in correct order
    df : pd.DataFrame
        Data we want to order

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with ordered feature
    """
    return df[feature_order]

def cchvae_method(counterfactual):
    """
    Function that returns C-CHVAE with respect to instance of interest x
    """

    def sampling(settings, types_dict, types_dict_c, out, ncounterfactuals, clf, n_batches_train, n_samples_train, k, n_input, degree_active):

        argvals = settings.split()
        args = Helpers.getArgs(argvals)

    

