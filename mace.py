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
from mace_modelConversion import forest2formula, mlp2formula
import mace_normalizedDistance

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

    def getPlausibilityFormula(model_symbols, dataset_obj, factual_sample, approach_string):
        # here is where the user specifies the following:
        #  1. data range plausibility
        #  2. data type plausibility
        #  3. actionability + mutability
        #  4. causal consistency

        ##############################################################################
        ## 1. data range plausibility
        ##############################################################################
        range_plausibility_counterfactual = And([And(GE(model_symbols['counterfactual'][attr_name_kurz]['symbol'], model_symbols['counterfactual'][attr_name_kurz]['lower_bound']),
            LE(model_symbols['counterfactual'][attr_name_kurz]['symbol'], model_symbols['counterfactual'][attr_name_kurz]['upper_bound'])) for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz')])
        range_plausibility_interventional = And([And(GE(model_symbols['interventional'][attr_name_kurz]['symbol'], model_symbols['interventional'][attr_name_kurz]['lower_bound']),
            LE(model_symbols['interventional'][attr_name_kurz]['symbol'], model_symbols['interventional'][attr_name_kurz]['upper_bound'])) for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz')])

        # IMPORTANT: a weird behavior of print(get_model(formula)) is that if there is
        #            a variable that is defined as a symbol, but is not constrained in
        #            the formula, then print(.) will not print the "verifying" value of
        #            that variable (as it can be anything). Therefore, we always use
        #            range plausibility constraints on ALL variables (including the
        #            interventional variables, even though they are only used for MINT
        #            and not MACE). TODO: find alternative method to print(model).
        range_plausibility = And([range_plausibility_counterfactual, range_plausibility_interventional])

        ##############################################################################
        ## 2. data type plausibility
        ##############################################################################
        onehot_categorical_plausibility = TRUE() # plausibility of categorical (sum = 1)
        onehot_ordinal_plausibility = TRUE() # plausibility ordinal (x3 >= x2 & x2 >= x1)

        if dataset_obj.is_one_hot:
            dict_of_siblings_kurz = dataset_obj.getDictOfSiblings('kurz')
            for parent_name_kurz in dict_of_siblings_kurz['cat'].keys():

                onehot_categorical_plausibility = And(onehot_categorical_plausibility, And(EqualsOrIff(Plus([model_symbols['counterfactual'][attr_name_kurz]['symbol']
                    for attr_name_kurz in dict_of_siblings_kurz['cat'][parent_name_kurz]]), Int(1))),
                    And(EqualsOrIff(Plus([model_symbols['interventional'][attr_name_kurz]['symbol']
                    for attr_name_kurz in dict_of_siblings_kurz['cat'][parent_name_kurz]]), Int(1))))

            for parent_name_kurz in dict_of_siblings_kurz['ord'].keys():

                onehot_ordinal_plausibility = And(onehot_ordinal_plausibility, And([GE(ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx]]['symbol']),
                    ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx + 1]]['symbol']))
                    for symbol_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1)]),
                    And([GE(ToReal(model_symbols['interventional'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx]]['symbol']),
                    ToReal(model_symbols['interventional'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx + 1]]['symbol']))
                    for symbol_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1)]))

        ##############################################################################
        ## 3. actionability + mutability
        #    a) actionable and mutable: both interventional and counterfactual value can change
        #    b) non-actionable but mutable: interventional value cannot change, but counterfactual value can
        #    c) immutable and non-actionable: neither interventional nor counterfactual value can change
        ##############################################################################
        actionability_mutability_plausibility = []
        for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
            attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]

            # a) actionable and mutable: both interventional and counterfactual value can change
            if attr_obj.mutability == True and attr_obj.actionability != 'none':

                if attr_obj.actionability == 'same-or-increase':
                    actionability_mutability_plausibility.append(GE(model_symbols['counterfactual'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                    actionability_mutability_plausibility.append(GE(model_symbols['interventional'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                elif attr_obj.actionability == 'same-or-decrease':
                    actionability_mutability_plausibility.append(LE(model_symbols['counterfactual'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                    actionability_mutability_plausibility.append(LE(model_symbols['interventional'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                elif attr_obj.actionability == 'any':
                    continue

            # b) mutable but non-actionable: interventional value cannot change, but counterfactual value can
            elif attr_obj.mutability == True and attr_obj.actionability == 'none':

            # IMPORTANT: when we are optimizing for nearest CFE, we completely ignore
            #            the interventional symbols, even though they are defined. In
            #            such a world, we also don't have any assumptions about the
            #            causal structure, and therefore, causal_consistency = TRUE()
            #            later in the code. Therefore, a `mutable but actionable` var
            #            (i.e., a variable that can change due to it's ancerstors) does
            #            not even exist. Thus, non-actionable variables are supported
            #            by restricing the counterfactual symbols.
            # TODO: perhaps a better way to structure this code is to completely get
            #       rid of interventional symbols when calling genSATExp.py with MACE.
                if 'mace' in approach_string:
                    actionability_mutability_plausibility.append(EqualsOrIff(model_symbols['counterfactual'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                elif 'mint' in approach_string:
                    actionability_mutability_plausibility.append(EqualsOrIff(model_symbols['interventional'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))

            # c) immutable and non-actionable: neither interventional nor counterfactual value can change
            else:

                actionability_mutability_plausibility.append(EqualsOrIff(model_symbols['counterfactual'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))
                actionability_mutability_plausibility.append(EqualsOrIff(model_symbols['interventional'][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz]))

        actionability_mutability_plausibility = And(actionability_mutability_plausibility)

        ##############################################################################
        ## 4. causal consistency
        ##############################################################################
        causal_consistency = TRUE()
        return And(range_plausibility, onehot_categorical_plausibility, onehot_ordinal_plausibility, actionability_mutability_plausibility, causal_consistency)

    def getDistanceFormula(model_symbols, dataset_obj, factual_sample, norm_type, approach_string, norm_threshold):

        variable_to_compute_distance_on = 'counterfactual'

        def getAbsoluteDifference(symbol_1, symbol_2):
            return Ite(GE(Minus(ToReal(symbol_1), ToReal(symbol_2)), Real(0)), Minus(ToReal(symbol_1), ToReal(symbol_2)), Minus(ToReal(symbol_2), ToReal(symbol_1)))

        # normalize this feature's distance by dividing the absolute difference by the
        # range of the variable (only applies for non-hot variables)
        normalized_absolute_distances = []
        normalized_squared_distances = []

        # IMPORTANT CHANGE IN CODE (Feb 04, 2020): prior to today, actionable/mutable
        # features overlapped. Now that we have introduced 3 types of variables
        # (actionable and mutable, non-actionable but mutable, immutable and non-actionable),
        # we must re-write the distance function to depent on all mutable features only,
        # while before we wrote distance as a function over actionable/mutable features.

        mutable_attributes = dataset_obj.getMutableAttributeNames('kurz')
        one_hot_attributes = dataset_obj.getOneHotAttributesNames('kurz')
        non_hot_attributes = dataset_obj.getNonHotAttributesNames('kurz')

        # 1. mutable & non-hot
        for attr_name_kurz in np.intersect1d(mutable_attributes, non_hot_attributes):
            normalized_absolute_distances.append(Div(ToReal(getAbsoluteDifference(model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'], factual_sample[attr_name_kurz])),
                ToReal(model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] - model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound'])))
            normalized_squared_distances.append(Pow(Div(ToReal(Minus(model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'], factual_sample[attr_name_kurz])),
                ToReal(model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] - model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound'])),
                Real(2)))

        # 2. mutable & integer-based & one-hot
        already_considered = []
        for attr_name_kurz in np.intersect1d(mutable_attributes, one_hot_attributes):
            if attr_name_kurz not in already_considered:
                siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
                if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
                    normalized_absolute_distances.append(Ite(And([EqualsOrIff(model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],factual_sample[attr_name_kurz])
                        for attr_name_kurz in siblings_kurz]),Real(0),Real(1)))
                    normalized_squared_distances.append(normalized_absolute_distances[-1])

                elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
                    normalized_absolute_distances.append(Div(ToReal(getAbsoluteDifference(Plus([model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                            for attr_name_kurz in siblings_kurz]), Plus([factual_sample[attr_name_kurz] for attr_name_kurz in siblings_kurz]),)),Real(len(siblings_kurz))))
                    normalized_squared_distances.append(Pow(Div(ToReal(Minus(Plus([model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                            for attr_name_kurz in siblings_kurz]),Plus([factual_sample[attr_name_kurz] for attr_name_kurz in siblings_kurz]),)),Real(len(siblings_kurz))),Real(2)))
                else:
                    raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
                already_considered.extend(siblings_kurz)

        # # 3. compute normalized squared distances
        # # pysmt.exceptions.SolverReturnedUnknownResultError
        # normalized_squared_distances = [
        #   # Times(distance, distance)
        #   Pow(distance, Int(2))
        #   for distance in normalized_absolute_distances
        # ]
        # # TODO: deprecate?
        # # def getSquaredifference(symbol_1, symbol_2):
        # #   return Times(
        # #     ToReal(Minus(ToReal(symbol_1), ToReal(symbol_2))),
        # #     ToReal(Minus(ToReal(symbol_2), ToReal(symbol_1)))
        # #   )


        # 4. sum up over everything allowed...
        # We use 1 / len(normalized_absolute_distances) below because we only consider
        # those attributes that are mutable, and for each sibling-group (ord, cat)
        # we only consider 1 entry in the normalized_absolute_distances
        if norm_type == 'zero_norm':
            distance_formula = LE(Times(Real(1 / len(normalized_absolute_distances)),Plus([Ite(Equals(elem,Real(0)),Real(0),Real(1)) for elem in normalized_absolute_distances])),
            Real(norm_threshold))
        elif norm_type == 'one_norm':
            distance_formula = LE(Times(Real(1 / len(normalized_absolute_distances)),ToReal(Plus(normalized_absolute_distances))),
            Real(norm_threshold))
        elif norm_type == 'two_norm':
            distance_formula = LE(Times(Real(1 / len(normalized_squared_distances)),ToReal(Plus(normalized_squared_distances))),Pow(Real(norm_threshold),Real(2)))
        elif norm_type == 'infty_norm':
            distance_formula = LE(Times(Real(1 / len(normalized_absolute_distances)),ToReal(Max(normalized_absolute_distances))),Real(norm_threshold))
        else:
            raise Exception(f'{norm_type} not recognized as a valid `norm_type`.')
        return distance_formula

    def getPrettyStringForSampleDictionary(sample, dataset_obj):
        if len(sample.keys()) == 0 :
            return 'No sample found.'
        key_value_pairs_with_x_in_key = {}
        key_value_pairs_with_y_in_key = {}
        for key, value in sample.items():
            if key in dataset_obj.getInputAttributeNames('kurz'):
                key_value_pairs_with_x_in_key[key] = value
            elif key in dataset_obj.getOutputAttributeNames('kurz'):
                key_value_pairs_with_y_in_key[key] = value
            else:
                raise Exception('Sample keys may only be `x` or `y`.')
        assert \
            len(key_value_pairs_with_y_in_key.keys()) == 1, \
            f'expecting only 1 output variables, got {len(key_value_pairs_with_y_in_key.keys())}'
        all_key_value_pairs = []
        for key, value in sorted(key_value_pairs_with_x_in_key.items(), key = lambda x: int(x[0][1:].split('_')[0])):
            all_key_value_pairs.append(f'{key} : {value}')
        all_key_value_pairs.append(f"{'y'}: {key_value_pairs_with_y_in_key['y']}")
        return f"{{{', '.join(all_key_value_pairs)}}}"

    def getDictSampleFromPySMTSample(pysmt_sample, dataset_obj):
        dict_sample = {}
        for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
            attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
            try:
                if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
                    dict_sample[attr_name_kurz] = bool(str(pysmt_sample[attr_name_kurz]) == 'True')
                elif attr_obj.attr_type == 'numeric-real':
                    dict_sample[attr_name_kurz] = float(eval(str(pysmt_sample[attr_name_kurz])))
                else: # refer to loadData.VALID_ATTRIBUTE_TYPES
                    dict_sample[attr_name_kurz] = int(str(pysmt_sample[attr_name_kurz]))
            except:
                raise Exception(f'Failed to read value from pysmt sample. Debug me manually.')
        return dict_sample       

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
        
        def assertPrediction(dict_sample, model_trained, dataset_obj):
            vectorized_sample = []
            for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
                vectorized_sample.append(dict_sample[attr_name_kurz])

            sklearn_prediction = int(model_trained.predict([vectorized_sample])[0])
            pysmt_prediction = int(dict_sample['y'])
            factual_prediction = int(factual_sample['y'])

            # IMPORTANT: sometimes, MACE does such a good job, that the counterfactual
            #            ends up super close to (if not on) the decision boundary; here
            #            the label is underfined which causes inconsistency errors
            #            between pysmt and sklearn. We skip the assert at such points.
            class_predict_proba = model_trained.predict_proba([vectorized_sample])[0]
            if np.abs(class_predict_proba[0] - class_predict_proba[1]) < 1e-10:
                return

            assert sklearn_prediction == pysmt_prediction, 'Pysmt prediction does not match sklearn prediction.'
            assert sklearn_prediction != factual_prediction, 'Counterfactual and factual samples have the same prediction.'

        factual_pysmt_sample = getPySMTSampleFromDictSample(factual_sample, dataset_obj)
    
        norm_lower_bound = 0
        norm_upper_bound = 1
        curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)

        model_formula = getModelFormula(model_symbols, model_trained)
        counterfactual_formula = getCounterfactualFormula(model_symbols, factual_pysmt_sample)
        plausibility_formula = getPlausibilityFormula(model_symbols, dataset_obj, factual_pysmt_sample, approach_string)
        distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, curr_norm_threshold)
        diversity_formula = TRUE() # simply initialize and modify later as new counterfactuals come in

        iters = 1
        max_iters = 3 #100
        counterfactuals = [] # list of tuples (samples, distances)
        # In case no counterfactuals are found (this could happen for a variety of
        # reasons, perhaps due to non-plausibility), return a template counterfactual
        counterfactuals.append({
            'counterfactual_sample': {},
            'counterfactual_distance': np.infty,
            'interventional_sample': {},
            'interventional_distance': np.infty,
            'time': np.infty,
            'norm_type': norm_type})

        print('Solving (not searching) for closest counterfactual using various distance thresholds...')

        while iters < max_iters and norm_upper_bound - norm_lower_bound >= epsilon:

            print(f'\tIteration #{iters:03d}: testing norm threshold {curr_norm_threshold:.6f} in range [{norm_lower_bound:.6f}, {norm_upper_bound:.6f}]...\t', end = '')
            iters = iters + 1

            formula = And(model_formula, counterfactual_formula, plausibility_formula, distance_formula, diversity_formula)

            solver_name = "z3"
            with Solver(name=solver_name) as solver:
                solver.add_assertion(formula)

                iteration_start_time = time.time()
                solved = solver.solve()
                iteration_end_time = time.time()

                if solved: # joint formula is satisfiable
                    model = solver.get_model()
                    print('solution exists & found.')
                    counterfactual_pysmt_sample = {}
                    interventional_pysmt_sample = {}
                    for (symbol_key, symbol_value) in model:
                    # symbol_key may be 'x#', {'p0#', 'p1#'}, 'w#', or 'y'
                        tmp = str(symbol_key)
                        if 'counterfactual' in str(symbol_key):
                            tmp = tmp[:-15]
                            if tmp in dataset_obj.getInputOutputAttributeNames('kurz'):
                                counterfactual_pysmt_sample[tmp] = symbol_value
                        elif 'interventional' in str(symbol_key):
                            tmp = tmp[:-15]
                            if tmp in dataset_obj.getInputOutputAttributeNames('kurz'):
                                interventional_pysmt_sample[tmp] = symbol_value
                        elif tmp in dataset_obj.getInputOutputAttributeNames('kurz'): # for y variable
                            counterfactual_pysmt_sample[tmp] = symbol_value
                            interventional_pysmt_sample[tmp] = symbol_value

                    # Convert back from pysmt_sample to dict_sample to compute distance and save
                    counterfactual_sample  = getDictSampleFromPySMTSample(counterfactual_pysmt_sample, dataset_obj)
                    interventional_sample  = getDictSampleFromPySMTSample(interventional_pysmt_sample, dataset_obj)

                    # Assert samples have correct prediction label according to sklearn model
                    assertPrediction(counterfactual_sample, model_trained, dataset_obj)
                    # of course, there is no need to assertPrediction on the interventional_sample

                    counterfactual_distance = mace_normalizedDistance.getDistanceBetweenSamples(
                    factual_sample, counterfactual_sample, norm_type, dataset_obj)
                    interventional_distance = mace_normalizedDistance.getDistanceBetweenSamples(
                    factual_sample, interventional_sample, norm_type, dataset_obj)
                    counterfactual_time = iteration_end_time - iteration_start_time
                    counterfactuals.append({
                    'counterfactual_sample': counterfactual_sample,
                    'counterfactual_distance': counterfactual_distance,
                    'interventional_sample': interventional_sample,
                    'interventional_distance': interventional_distance,
                    'time': counterfactual_time,
                    'norm_type': norm_type})

                    # Update diversity and distance formulas now that we have found a solution
                    # TODO: I think the line below should be removed, because in successive
                    #       reductions of delta, we should be able to re-use previous CFs
                    # diversity_formula = And(diversity_formula, getDiversityFormulaUpdate(model))

                    # IMPORTANT: something odd happens somtimes if use vanilla binary search;
                    #            On the first iteration, with [0, 1] bounds, we may see a CF at
                    #            d = 0.22. When we update the bounds to [0, 0.5] bounds,  we
                    #            sometimes surprisingly see a new CF at distance 0.24. We optimize
                    #            the binary search to solve this.
                    norm_lower_bound = norm_lower_bound
                    # norm_upper_bound = curr_norm_threshold
                    norm_upper_bound = float(counterfactual_distance + epsilon / 100) # not float64
                    curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
                    distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, curr_norm_threshold)

                else: # no solution found in the assigned norm range --> update range and try again
                    with Solver(name=solver_name) as neg_solver:
                        neg_formula = Not(formula)
                        neg_solver.add_assertion(neg_formula)
                        neg_solved = neg_solver.solve()
                        if neg_solved:
                            print('no solution exists.')
                            norm_lower_bound = curr_norm_threshold
                            norm_upper_bound = norm_upper_bound
                            curr_norm_threshold = getCenterNormThresholdInRange(norm_lower_bound, norm_upper_bound)
                            distance_formula = getDistanceFormula(model_symbols, dataset_obj, factual_pysmt_sample, norm_type, approach_string, curr_norm_threshold)
                        else:
                            print('no solution found (SMT issue).')
                            quit()
                            break

        closest_counterfactual_sample = sorted(counterfactuals, key = lambda x: x['counterfactual_distance'])[0]
        closest_interventional_sample = sorted(counterfactuals, key = lambda x: x['interventional_distance'])[0]
        return counterfactuals, closest_counterfactual_sample, closest_interventional_sample
    
    def genExp(model_trained, dataset_obj, factual_sample, norm_type, approach_string, epsilon):

        start_time = time.time()
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
        end_time = time.time()
        
        # print('\n')
        # print(f"Factual sample: \t\t {getPrettyStringForSampleDictionary(factual_sample, dataset_obj)}")
        # if 'mace' in approach_string:
        #     print(f"Nearest counterfactual sample:\t {getPrettyStringForSampleDictionary(closest_counterfactual_sample['counterfactual_sample'], dataset_obj)} (verified)")
        #     print(f"Minimum counterfactual distance: {closest_counterfactual_sample['counterfactual_distance']:.6f}")
        
        return {'cf_found': True, 'cf_plausible': True, 'cf_time': end_time - start_time,
        'cf_sample': closest_counterfactual_sample['counterfactual_sample'],
        'cf_distance': closest_counterfactual_sample['counterfactual_distance']}
    
    model_trained = counterfactual.model.model
    dataset_obj = counterfactual.data
    # factual_sample = counterfactual.ioi.normal_x_df.T.to_dict()[counterfactual.ioi.normal_x_df.index[0]]
    factual_sample = counterfactual.ioi.normal_x_df.T.to_dict()[counterfactual.ioi.normal_x_df.index[0]]
    factual_sample['y'] = bool(counterfactual.ioi.label)
    norm_type = 'two_norm' # Possible values: zero_norm, one_norm, two_norm, infty_norm
    epsilon = 1e-1

    results_dict = genExp(model_trained, dataset_obj, factual_sample, norm_type, 'mace', epsilon)
    cf, run_time = results_dict['cf_sample'], results_dict['cf_time']
    return cf, run_time

    
    