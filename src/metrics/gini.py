from typing import Dict
import numpy as np


def gini_feature(feature: str, dataset: Dict[str, np.ndarray], feature_to_predict: np.ndarray) -> float:
    """

    Args:
        feature:
        dataset:
        feature_to_predict

    Returns:

    """
    element_feature: np.array = dataset[feature]
    gini_index: Dict[str, float] = {}
    unique, counts = np.unique(element_feature, return_counts=True)
    unique_dict = dict(zip(unique, counts))
    length_data = len(dataset)

    for attribute in element_feature:
        gini_index[attribute] = gini_attribute(dataset=dataset, attribute=attribute, feature=feature,
                                               feature_to_predict=feature_to_predict)
    weighted_sum: float = 0.0
    for attribute, gini in gini_index.items():
        weighted_sum += (unique_dict[attribute] / length_data) * gini

    # gini_index['weighted_sum'] = weighted_sum
    return weighted_sum


def gini_attribute(dataset: Dict[str, np.ndarray],  attribute: str, feature: str, feature_to_predict: np.ndarray):
    """
    Compute gini index for every attribute of a feature
    This is a utility function used in gini_feature
    Args:
        dataset: DataSet Object
        attribute:
        feature:
        feature_to_predict:

    Returns:

    """
    element_feature: np.array = dataset[feature]
    unique: np.array = np.unique(feature_to_predict)
    dict_gini: Dict[str, int] = {}

    for i in range(len(unique)):
        dict_gini[unique[i]] = 0
        for j in range(len(element_feature)):
            if element_feature[j] == attribute and feature_to_predict[i] == unique[i]:
                dict_gini[unique[i]] += 1

    gini: np.array = np.array(list(dict_gini.values()))
    nb_occurence_attribute = np.sum(gini)
    gini = gini / nb_occurence_attribute
    return 1 - np.sum(gini ** 2)
