"""
Model-Agnostic Counterfactual Explanations (MACE)
Based on original authors implementation: Please see https://github.com/amirhk/mace
"""

"""
Imports
"""
import numpy as np
import pandas as pd
import time
from pysmt.shortcuts import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from modelConversion import forest2formula, mlp2formula

class MACE:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = mace_method(counterfactual)

def mace_method(counterfactual):
    """
    Function that returns MACE with respect to instance of interest x
    """
    def getModelFormula(model_symbols, model_trained):
            if isinstance(model_trained, RandomForestClassifier):
                model2formula = lambda a,b : forest2formula(a,b)
            elif isinstance(model_trained, MLPClassifier):
                model2formula = lambda a,b : mlp2formula(a,b)
            return model2formula(model_trained, model_symbols)

    def getCounterfactualFormula(model_symbols, factual_sample):
        return EqualsOrIff(model_symbols['output']['y']['symbol'], Not(factual_sample['y']))

    def genExp(explanation_file_name, model_trained, dataset_obj, factual_sample, norm_type,
               approach_string, epsilon):

        model_symbols = {'counterfactual': {}, 'interventional': {}, 'output': {'y': {'symbol': Symbol('y', BOOL)}}}

        for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
            attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
            lower_bound = attr_obj.lower_bound
            upper_bound = attr_obj.upper_bound
            # print(f'\n attr_name_kurz: {attr_name_kurz} \t\t lower_bound: {lower_bound} \t upper_bound: {upper_bound}', file = log_file)
            if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
                continue # do not overwrite the output
            if attr_obj.attr_type == 'numeric-real':
                model_symbols['counterfactual'][attr_name_kurz] = {
                    'symbol': Symbol(attr_name_kurz + '_counterfactual', REAL),
                    'lower_bound': Real(float(lower_bound)),
                    'upper_bound': Real(float(upper_bound))}
                model_symbols['interventional'][attr_name_kurz] = {
                    'symbol': Symbol(attr_name_kurz + '_interventional', REAL),
                    'lower_bound': Real(float(lower_bound)),
                    'upper_bound': Real(float(upper_bound))}
            else: # refer to loadData.VALID_ATTRIBUTE_TYPES
                model_symbols['counterfactual'][attr_name_kurz] = {
                    'symbol': Symbol(attr_name_kurz + '_counterfactual', INT),
                    'lower_bound': Int(int(lower_bound)),
                    'upper_bound': Int(int(upper_bound))}
                model_symbols['interventional'][attr_name_kurz] = {
                    'symbol': Symbol(attr_name_kurz + '_interventional', INT),
                    'lower_bound': Int(int(lower_bound)),
                    'upper_bound': Int(int(upper_bound))}

        all_counterfactuals, closest_counterfactual_sample, closest_interventional_sample = findClosestCounterfactualSample(model_trained, model_symbols, dataset_obj, factual_sample, norm_type,
                                                                                                                            approach_string, epsilon)

    def getPySMTSampleFromDictSample(dict_sample, dataset_obj):
        pysmt_sample = {}
        for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
            attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
            if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
                pysmt_sample[attr_name_kurz] = Bool(dict_sample[attr_name_kurz])
            elif attr_obj.attr_type == 'numeric-real':
                pysmt_sample[attr_name_kurz] = Real(float(dict_sample[attr_name_kurz]))
            else:
                pysmt_sample[attr_name_kurz] = Int(int(dict_sample[attr_name_kurz]))
        return pysmt_sample

    def findClosestCounterfactualSample(model_trained, model_symbols, dataset_obj, factual_sample, norm_type, approach_string, epsilon):

        def getCenterNormThresholdInRange(lower_bound, upper_bound):
            return (lower_bound + upper_bound) / 2
        
        factual_pysmt_sample = getPySMTSampleFromDictSample(factual_sample, dataset_obj)
    
        norm_lower_bound = 0
        norm_upper_bound = 1
        curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)

        model_formula = getModelFormula(model_symbols, model_trained)
        counterfactual_formula = getCounterfactualFormula(model_symbols, factual_pysmt_sample)
        plausibility_formula = getPlausibilityFormula(model_symbols, dataset_obj, factual_pysmt_sample, approach_string)
        distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, curr_norm_threshold)
        diversity_formula = TRUE() # simply initialize and modify later as new counterfactuals come in
