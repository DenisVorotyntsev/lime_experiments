from typing import List, Dict, Tuple, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from lime.lime_tabular import LimeTabularExplainer


def explain_point(
        data: pd.DataFrame,
        cat_columns: List[str],
        model,
        ind_to_explain: int = 10,
        tabular_params: Dict[str, any] = None,
        explainer_params: Dict[str, any] = None,
        verbose: bool = False
):
    """

    :param data: train dataframe
    :param cat_columns: list of categorical columns
    :param model: trained model
    :param ind_to_explain: index of data point in train dataframe to explain
    :param tabular_params: LimeTabularExplainer kwargs
    :param explainer_params: LimeTabularExplainer.explain_instance() kwargs
    :param verbose: show visualisation of explanations
    :return:
    """
    if tabular_params is None:
        tabular_params = {}
    if explainer_params is None:
        explainer_params = {}

    explainer = LimeTabularExplainer(
        np.array(data),
        feature_names=list(data.columns),
        categorical_features=cat_columns,
        **tabular_params
    )

    explanation = explainer.explain_instance(
        data.loc[ind_to_explain],
        model.predict_proba,
        num_features=data.shape[1],
        **explainer_params
    )
    if verbose:
        explanation.show_in_notebook(show_table=False, show_all=False)
    return explanation


def generate_explanations_in_df(
        explanation,
        columns: List[str]
) -> pd.DataFrame:
    """
    Transform lime explanations to dataframe
    :param explanation: lime explanations results
    :param columns: train dataframe columns
    :return:
    """
    explanations_dict = {
        columns[column_ind]: explanation_weight for column_ind, explanation_weight in explanation.local_exp[1]
    }
    for col in [col for col in columns if col not in explanations_dict.keys()]:
        explanations_dict[col] = 0
    df = pd.DataFrame(explanations_dict, index=[0])
    df = df[columns]
    return df


def run_several_hparams_explanations(
        data: pd.DataFrame,
        ind: int,
        explain_point_kwargs: Dict[str, any],
        params: Tuple[Dict[str, any], Dict[str, any]],
        verbose: bool = False
) -> pd.DataFrame:
    """

    :param data: train dataframe
    :param ind: index of train dataframe to make explanations to
    :param explain_point_kwargs: dict with "cat_columns" and "model" keys
    :param params: params of explanations to check
    :param verbose: verbose progress bar
    :return:
    """
    explanation_dfs = []
    for params in tqdm(params, disable=~verbose):
        explanation = explain_point(
            data=data,
            ind_to_explain=ind,
            tabular_params=params[0],
            explainer_params=params[1],
            **explain_point_kwargs
        )
        explanation_df = generate_explanations_in_df(explanation, data.columns)
        explanation_dfs.append(explanation_df)
    explanations = pd.concat(explanation_dfs, axis=0).reset_index(drop=True)
    return explanations


def calculate_explanations_stats(
        explanations: pd.DataFrame,
        features_to_use: Union[str, List[str]] = "all"
) -> float:
    """
    Calculate measure of lime explanations goodness calculated using different hyperparameters.
    Measure is defined as mean of pairwise Spearman's rank correlation between features weights calculated using
    different set of hyperparameters.
    :param explanations: dataframe with features weights using different set of hyperparameters
    :param features_to_use: "all" for all features or list of features to use for measuring
    :return: measure
    """
    # transpose explanations dataframe.
    # In transposed dataframe index - names of features, columns - feature weights (importances)
    explanations_ = explanations.T
    if type(features_to_use) == str:
        if features_to_use == "all":
            explanations_ = explanations_[explanations_.columns]
        else:
            raise Exception("Not expected value in features_to_use")
    elif type(features_to_use) == list:
        explanations_ = explanations_.loc[features_to_use]
    else:
        raise Exception("Not expected value in features_to_use")

    # calculate pairwise correlations
    corr = spearmanr(explanations_)[0]

    # replace diagonal values with nan
    diag_idx = (np.arange(corr.shape[0]), np.arange(corr.shape[0]))
    corr[diag_idx] = np.nan

    # get values for the lower-triangle
    iu = np.tril_indices(corr.shape[0])

    # calculate mean of those values
    result = np.nanmean(corr[iu])
    return result
