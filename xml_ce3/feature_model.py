from functools import reduce
from typing import Union, Callable, List
import copy 
import pandas as pd 
import patsy
from sklearn.model_selection import train_test_split as train_test_split_fun

quiet=False
def print_self(method):
    def wrapped_method(self, *args, **kwargs):
        output = method(self, *args, **kwargs)
        if not quiet:
            print(self)
        return output
    return wrapped_method

class FeatureModel:
    def __init__(self, path_to_csv, include_intercept_feature=False):
        """Create a new modeling object, target ~ featureA + featureB + ...

        Args:
            path_to_csv (str): Relative or absolute path to .csv-file that contains the dataset
            include_intercept_feature (bool, optional): Whether or not to include a constant feature. Defaults to False.
        """
        self.path_to_csv = path_to_csv
        self.df = pd.read_csv(path_to_csv)
        self._formula = []
        self._target = None
        self.intercept = include_intercept_feature
        self._funs = {}

    def reset(self, keep_target=True):
        """Reset the modeling object.

        Args:
            keep_target (bool, optional): Keeps the target. Defaults to True.
        """
        self.df = pd.read_csv(self.path_to_csv)
        self._formula = []
        if not keep_target:
            self._target = None

    @print_self
    def undo_add_operation(self):
        """Undo the last `.add_feature` or `.add_function_feature`
        """
        self._formula = self._formula[:-1]

    def __repr__(self):
        try:
            return f"FeatureModel({self._reduce_list(self.target+self.formula)})"
        except:
            return "FeatureModel(->Not fully specified yet!)"

    @property
    def target(self):
        if self._target is None:
            raise Exception("Use `.add_target` to add a target variable first")
        return self._target

    @property
    def formula(self):
        if len(self._formula) == 0:
            raise Exception("Use `.add_feature` or `.add_function_feature` to add a feature variable first")
        return ["1 "] + self._formula if self.intercept else ["-1 "] + self._formula
    
    def return_Xy(self, transform_before=None, transform_after=None, train_test_split=False):
        """Return a tuple of X,y-arrays. Internal logic is:\\
            1. df = copy.copy(self.df) # never modifiy the underlying dataframe\\
            2. df = transform_before(df) # or skip if `transform_before` is `None`\\
            3. X,y = function(self) # according to the rules of this modeling object, target ~ featureA + featureB + ...\\
            4. X,y = transform_after(X,y) # or skip if `transform_after` is `None`\\
            5. X_train, y_train, X_val, y_val = splitting_function(X,y) # or skip if `train_test_split` is False\\

        Args:
            transform_before (Callable[pd.DataFrame, pd.DataFrame], optional): Transformation that maps the dataframe-object to a dataframe-object. Defaults to None.
            transform_after (Callable[[np.ndarray, np.ndarray], [np.ndarray, np.ndarray]], optional): Transformation that maps X,y to X,y   . Defaults to None.
            train_test_split (bool, optional): If True return `X_train, y_train, X_val, y_val` instead where 20% of data is for validation. Defaults to False.

        Returns:
            Tuple: Tuple of arrays
        """
        
        # define all functions in this local namespace
        for f_name, f in self._funs.items():
            locals()[f_name] = f
        
        df = copy.copy(self.df)
        if transform_before:
            df = transform_before(df)

        y, X = patsy.dmatrices(self._reduce_list(self.target+self.formula), df, return_type="dataframe")
        
        if transform_after:
            X, y = transform_after(X, y)
            
        # this is technically not entirely correct (that is: to apply transforms before splitting) but it doesn't matter for us
        if train_test_split:
            X_train, X_val, y_train, y_val = \
            train_test_split_fun(X, y, test_size=0.2, random_state=1)
            return X_train, y_train, X_val, y_val
        else:
            return X, y

    @print_self
    def add_function_feature(self, 
        new_feature: Union[
            Callable[[float], float],
            Callable[[float,float], float],
            Callable[[float,float,float], float]
        ], 
        df_column_name: str, 
        second_andOr_third_function_argument: Union[
            List[Union[str, float]],
            List[List[Union[str, float]]]
        ] = []
        ):
        """Add a new feature to the modeling object. The new feature's value is the datapoint-wise evaluation of the function `new_feature` using one or multiple columns of the dataframe-object.

        Args:
            function (Union[ Callable[[float], float], Callable[[float,float], float], Callable[[float,float,float], float] ]): The function used to create the new feature
            df_column_name (str): The datapoint-wise numerical value of this column of the dataframe-object will be available in the function `new_feature` through the first argument.
            second_andOr_third_function_argument (Union[ List[Union[str, float]], List[List[Union[str, float]]] ], optional): 
                If length is 0, then `new_feature` has only one argument. If length is N, create N new features using the N different list elements. 
                If every list element is another list, then `new_feature` takes three arguments or more. Defaults to [].

        Examples:
            >>> def squared_temp(temp): return temp**2
            >>> data.add_function_feature(squared_temp, "temp")

            >>> def powers_of_temp(temp, power): return temp**power
            >>> data.add_function_feature(powers_of_temp, "temp", [2,3,4])

            >>> def interaction(f1, f2): return f1*f2
            >>> data.add_function_feature(interaction, "temp", ["workingday"])

            >>> def interaction(f1, f2, f2_value): return f1*(f2==f2_value)
            >>> data.add_function_feature(interaction, "temp", [["season", 1], ["season", 2], ["season", 3]])
        """
        fun = new_feature
        vs = second_andOr_third_function_argument

        self._funs[fun.__name__] = fun
        
        if vs == []:
            self._formula += [f"+ {fun.__name__}({df_column_name}) "]
            return 

        if isinstance(vs[0], list):
            vs = [self._reduce_list(v, ",") for v in vs]

        self._formula += [self._reduce_list([f" + {fun.__name__}({df_column_name},{v})" for v in vs])]

    @print_self
    def add_feature(self, df_column_name: str):
        """Add a new feature to the modeling object. 
        The new feature is create by directly using the respective dataframe-column as input / feature. For more sophisticated features use `add_function_feature` instead.

        Args:
            df_column_name (str): The column name of the data as specified in the dataframe-object.
        """
        self._formula += [f"+ {df_column_name} "]
        
    @print_self
    def add_all_features_but_target(self):
        """Add all dataframe columns as features
        """
        for feature in self.df.columns:
            if feature == self.target[0][:-3]:
                continue
            self._formula += [f"+ {feature} "]
        
    @print_self
    def add_target(self, df_column_name: str):
        """Add the target variable to the modeling object

        Args:
            df_column_name (str): The column name of the data as specified in the dataframe-object.
        """
        self._target = [f"{df_column_name} ~ "]

    @staticmethod
    def _reduce_list(l: List[Union[str, float]], delimiter="") -> str:
        l = [str(ele) for ele in l]
        long_str = reduce(lambda x,y:x+delimiter+y, l, "")
        # the first delimiter is too much
        return long_str[len(delimiter):]
