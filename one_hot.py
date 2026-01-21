import numpy as np
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple
# from pymatgen import Composition, Element # Old version of pymatgen in syn_gen_release env
from pymatgen.core.composition import Composition, Element # Updated for dqn env
# import chemml.chem.magpie_python as magpie
try:
    import chemml  # type: ignore
except ModuleNotFoundError:
    # chemml is only needed for a legacy, commented-out featurizer implementation.
    chemml = None

import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
from pymatgen.core.composition import Composition

max_num_steps = 5

# original
element_set = ['Te', 'Sc', 'C', 'Hg', 'Ru', 'Na', 'Co', 'Mo', 'I', 'Tm', 'F', 'Al', 'Pd', 'Fe', 'Th', 'Cs', 'Gd', 'W', 'Ta', 'Dy', 'Pb', 'Rb', 'Ba', 'Ce', 'Ga', 'Tl', 'Mn', 'B', 'Ni', 'Tb', 'Hf', 'Ge', 'V', 'Ho', 'In', 'Cd', 'Yb', 'Pt', 'Nd', 'Mg', 'Zr', 'Re', 'P', 'Sb', 'O', 'N', 'Zn', 'Au', 'Lu', 'Be', 'Cr', 'Ag', 'Pu', 'Si', 'Cu', 'Os', 'Li', 'Am', 'Pr', 'S', 'As', 'Ti', 'Nb', 'Eu', 'H', 'Br', 'La', 'Er', 'Sm', 'Cl', 'Sn', 'K', 'Sr', 'Rh', 'Se', 'U', 'Y', 'Bi', 'Ca', 'Ir']

# for oxides - intersection between oqmd-formation-energy and rf_sintering_T datasets
element_set = ['O', 'Te', 'N', 'B', 'Tm', 'Ga', 'Hf', 'Ca', 'Al', 'P', 'Li', 'S', 'Cr', 'Zr', 'Ta', 'Sn', 'Au', 'Hg', 'Cd', 'Mn', 'Cs', 'Pd', 'Th', 'K', 'Ti', 'Ag', 'Zn', 'W', 'Ce', 'Nd', 'Sr', 'Tl', 'Cl', 'Mg', 'Pr', 'Rb', 'Pb', 'Ru', 'Ho', 'Nb', 'Mo', 'C', 'V', 'Er', 'Pt', 'Fe', 'Ir', 'Sb', 'Y', 'Na', 'Co', 'Be', 'In', 'La', 'U', 'Pu', 'As', 'Sm', 'Br', 'Ni', 'Eu', 'Ba', 'F', 'Rh', 'Yb', 'Gd', 'Os', 'Lu', 'Ge', 'Cu', 'H', 'Sc', 'Si', 'Re', 'Dy', 'Bi', 'Tb', 'Se']

comp_set  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
step_set  = [x for x in range(1,max_num_steps+1)]


def _to_tuple(items: Sequence[str] | Iterable[str]) -> Tuple[str, ...]:
    return tuple(items)


@lru_cache(maxsize=64)
def _build_one_hot_mappings(items: Tuple[str, ...]) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[float, ...], str]]:
    item_to_oh: Dict[str, np.ndarray] = {}
    oh_to_item: Dict[Tuple[float, ...], str] = {}
    for idx, item in enumerate(items):
        enc = np.zeros(len(items))
        enc[idx] = 1.0
        item_to_oh[item] = enc
        oh_to_item[tuple(enc.tolist())] = item
    return item_to_oh, oh_to_item

def _get_target_char_sequence(compound_string):
    comp = Composition(compound_string).formula.replace(" ", "")
    char_seq = re.findall(r"[A-Z][a-z]?|[0-9]|.", comp)

    return char_seq
def onehot_target(target):
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    all_digits = [str(i) for i in range(0, 10)]
    target_charset = ["<NULL>"] + all_elements + all_digits + ["."]
    charset = ["<NULL>"] + all_elements + all_digits
    charset_size = len(charset)
    target_charset_size = len(target_charset)
    max_material_length = 14
    max_target_length = 40
    min_operation_length = 3
    max_operation_length = 20
    paper_batch_size = 50
    max_num_precs = 5
    try:
        mat_char_seq = _get_target_char_sequence(target)
        filtered_char_seq = [str(c) for c in mat_char_seq if str(c) in target_charset]
        char_seq_vec = np.zeros(shape=(max_target_length, target_charset_size))
        for j, char in enumerate(filtered_char_seq):
            char_vec = np.zeros(shape=(target_charset_size,))
            char_vec[target_charset.index(char)] = 1.0
            char_seq_vec[j] = char_vec
        for i in range(len(char_seq_vec), max_target_length):
            char_vec = np.zeros(shape=(target_charset_size,))
            char_vec[0] = 1.0
            char_seq_vec[i] = char_vec
    except Exception as e:
            print(target) 
    return char_seq_vec
# print(onehot_target('BaTiO3'))

# Version 1 of featurizer - no longer works with dqn env
# meredig = chemml.chem.magpie_python.MeredigAttributeGenerator()
# elem_frac = chemml.chem.magpie_python.ElementFractionAttributeGenerator()
# val_shell = chemml.chem.magpie_python.ValenceShellAttributeGenerator()
# charge_dep = chemml.chem.magpie_python.ChargeDependentAttributeGenerator()
# elem_prop = chemml.chem.magpie_python.ElementalPropertyAttributeGenerator()
# ionicity = chemml.chem.magpie_python.IonicityAttributeGenerator()
# stoichio = chemml.chem.magpie_python.StoichiometricAttributeGenerator()
# yang_omega = chemml.chem.magpie_python.YangOmegaAttributeGenerator()

# def featurize_target(target, feat_to_included = [
#                                                 'meredig',
#                                                 'elem_frac',
#                                                 'val_shell',
#                                                 'charge_dep',
#                                                 'elem_prop',
#                                                 'ionicity',
#                                                 'stoichio',
#                                                 'yang_omega',
#                                                 ]
#                     ): 
#     '''
#     Featurization of material using Magpie embeddings (https://hachmannlab.github.io/chemml/chemml.chem.magpie_python.html)

#     Arg:
#     target: Str.
#     feat_to_included: List (of Str). Features to be included. Default all.

#     Returns:
#     concat: np.array of concatenated features of the material
#     '''
#     chemical = magpie.CompositionEntry(composition=target)
#     features =  {}
#     features['meredig']    = meredig.generate_features(entries = [chemical])
#     features['elem_frac']  = elem_frac.generate_features(entries = [chemical])
#     features['val_shell']  = val_shell.generate_features(entries = [chemical])
#     features['charge_dep'] = charge_dep.generate_features(entries = [chemical])
#     features['elem_prop']  = elem_prop.generate_features(entries = [chemical])
#     features['ionicity']   = ionicity.generate_features(entries = [chemical])
#     features['stoichio']   = stoichio.generate_features(entries = [chemical])
#     features['yang_omega'] = yang_omega.generate_features(entries = [chemical])

#     concat = pd.concat([features[feat] for feat in feat_to_included], axis = 1)
#     return concat

# print(featurize_target('BaTiO3'))


# Version 2 of featurizer - works with dqn env
feature_calculators = MultipleFeaturizer([
    cf.element.Stoichiometry(),
    cf.composite.ElementProperty.from_preset("magpie"),
    cf.orbital.ValenceOrbital(props=["avg"]),
    cf.ion.IonProperty(fast=True)
])

def featurize_target(target, feature_calculator = feature_calculators): 
    '''
    Featurization of material using Magpie embeddings (https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.html)

    Arg:
    target: Str.

    Returns: 
    features: List of features of the material
    '''
    if target != '': # If not empty string
        chemical = Composition(target)
        features = feature_calculator.featurize(chemical)
        if 'compound possible' in feature_calculator.feature_labels(): # Encode 'compound possible' with 0 = False and 1 = True
            compound_poss = features[-3]
            if compound_poss == True:
                features[-3] = 1
            else:
                features[-3] = 0
    else: # empty string, starting state
        features = [0]*len(feature_calculator.feature_labels())
    return features
if __name__ == "__main__":
    print(featurize_target('Ba0.5Ti0.5O1.5'))


# Find element to one-hot dictionary
element_to_one_hot_dict = {}
for element_idx in range(len(element_set)):
    element = element_set[element_idx]
    enc = np.zeros(len(element_set))
    enc[element_idx] = 1
    element_to_one_hot_dict[element] = enc

# Find one-hot dictionary to element (inverse mapping)
one_hot_to_element_dict = {}
for element in element_to_one_hot_dict.keys():
    one_hot_to_element_dict[tuple(element_to_one_hot_dict[element])] = element # find the inverse mapping

def element_to_one_hot(elements):
    """
    converts a single element, or a list of multiple elements into their one-hot form

    Args:
    elements: List. list of elements

    Returns:
    element_to_one_hot: List of np.array each with shape (1, no. of elements in element_set)
    '''

    """
def element_to_one_hot(elements: Sequence[str], element_set_override: Sequence[str] | None = None):
    """Convert element symbol(s) to one-hot.

    Backward compatible: if element_set_override is None, uses global element_set.
    """
    if element_set_override is None:
        mapping = element_to_one_hot_dict
    else:
        mapping, _ = _build_one_hot_mappings(_to_tuple(element_set_override))

    out: List[np.ndarray] = []
    for element in elements:
        out.append(mapping[element])
    return out

def one_hot_to_element(one_hot_encs):
    """
    converts a single element, or a list of multiple elements in one-hot form into their string form

    Args:
    one_hot_encs: List. list of elements in one-hot form (tuple since dictionary accepts immutable keys)
    i.e. [ tuple(1,0,...,0,0),
           ...
           tuple(0,1,...,0,0)]

    Returns:
    one_hot_to_element: List of elements in string form
    """
def one_hot_to_element(one_hot_encs: Sequence[Sequence[float] | Tuple[float, ...]], element_set_override: Sequence[str] | None = None):
    """Convert one-hot vector(s) back to element symbols.

    Backward compatible: if element_set_override is None, uses global element_set.
    """
    if element_set_override is None:
        inverse = one_hot_to_element_dict
    else:
        _, inverse = _build_one_hot_mappings(_to_tuple(element_set_override))

    out: List[str] = []
    for enc in one_hot_encs:
        out.append(inverse[tuple(enc)])
    return out

########
# Find step to one-hot dictionary
step_to_one_hot_dict = {}
for step_idx in range(len(step_set)):
    step = step_set[step_idx]
    enc = np.zeros(len(step_set))
    enc[step_idx] = 1
    step_to_one_hot_dict[step] = enc

# Find one-hot dictionary to step (inverse mapping)
one_hot_to_step_dict = {}
for step in step_to_one_hot_dict.keys():
    one_hot_to_step_dict[tuple(step_to_one_hot_dict[step])] = step # find the inverse mapping

def step_to_one_hot(steps):
    """
    converts a single step, or a list of multiple steps into their one-hot form

    Args:
    steps: List. list of steps

    Returns:
    step_to_one_hot: List of np.array each with shape (1, no. of steps in step_set)
    '''

    """
def step_to_one_hot(steps: Sequence[int], step_set_override: Sequence[int] | None = None):
    """Convert step(s) to one-hot.

    Backward compatible: if step_set_override is None, uses global step_set.
    """
    if step_set_override is None:
        mapping = step_to_one_hot_dict
    else:
        items = tuple(str(x) for x in step_set_override)
        mapping_str, _ = _build_one_hot_mappings(items)
        mapping = {int(k): v for k, v in mapping_str.items()}

    out: List[np.ndarray] = []
    for step in steps:
        out.append(mapping[step])
    return out

def one_hot_to_step(one_hot_encs):
    """
    converts a single step, or a list of multiple steps in one-hot form into their string form

    Args:
    one_hot_encs: List. list of steps in one-hot form (tuple since dictionary accepts immutable keys)
    i.e. [ tuple(1,0,...,0,0),
           ...
           tuple(0,1,...,0,0)]

    Returns:
    one_hot_to_step: List of steps in string form
    """
def one_hot_to_step(one_hot_encs: Sequence[Sequence[float] | Tuple[float, ...]], step_set_override: Sequence[int] | None = None):
    """Convert one-hot vector(s) back to step index."""
    if step_set_override is None:
        inverse = one_hot_to_step_dict
    else:
        items = tuple(str(x) for x in step_set_override)
        _, inverse_str = _build_one_hot_mappings(items)
        inverse = {k: int(v) for k, v in inverse_str.items()}

    out: List[int] = []
    for enc in one_hot_encs:
        out.append(inverse[tuple(enc)])
    return out



########

# Find composition to one-hot dictionary
comp_to_one_hot_dict = {}
for comp_idx in range(len(comp_set)):
    comp = comp_set[comp_idx]
    enc = np.zeros(len(comp_set))
    enc[comp_idx] = 1
    comp_to_one_hot_dict[comp] = enc

# Find one-hot dictionary to composition (inverse mapping)
one_hot_to_comp_dict = {}
for comp in comp_to_one_hot_dict.keys():
    one_hot_to_comp_dict[tuple(comp_to_one_hot_dict[comp])] = comp # find the inverse mapping

def comp_to_one_hot(comps):
    """
    converts a single composition, or a list of multiple composition into their one-hot form

    Args:
    comps: List. list of compositions

    Returns:
    comp_to_one_hot: List of np.array each with shape (1, no. of compositions in comp_set)
    '''

    """
def comp_to_one_hot(comps: Sequence[str], comp_set_override: Sequence[str] | None = None):
    """Convert composition token(s) to one-hot.

    Backward compatible: if comp_set_override is None, uses global comp_set.
    """
    if comp_set_override is None:
        mapping = comp_to_one_hot_dict
    else:
        mapping, _ = _build_one_hot_mappings(_to_tuple(comp_set_override))

    out: List[np.ndarray] = []
    for comp in comps:
        out.append(mapping[comp])
    return out

def one_hot_to_comp(one_hot_encs):
    """
    converts a single composition, or a list of multiple compositions in one-hot form into their string form

    Args:
    one_hot_encs: List. list of comps in one-hot form (tuple since dictionary accepts immutable keys)
    i.e. [ tuple(1,0,...,0,0),
           ...
           tuple(0,1,...,0,0)]

    Returns:
    one_hot_to_comp: List of compositions in string form
    """
def one_hot_to_comp(one_hot_encs: Sequence[Sequence[float] | Tuple[float, ...]], comp_set_override: Sequence[str] | None = None):
    """Convert one-hot vector(s) back to composition tokens."""
    if comp_set_override is None:
        inverse = one_hot_to_comp_dict
    else:
        _, inverse = _build_one_hot_mappings(_to_tuple(comp_set_override))

    out: List[str] = []
    for enc in one_hot_encs:
        out.append(inverse[tuple(enc)])
    return out

# ======= Testing functions =======
# print(element_to_one_hot(['Te', 'C', 'Ru']))
# print(one_hot_to_element([(0., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.)]))
# print(comp_to_one_hot(['2', '9']))
# print(one_hot_to_comp([(0., 0., 1., 0., 0., 0., 0., 0., 0., 0.), (0., 0., 0., 0., 0., 0., 0., 0., 0., 1.)]))

# print(onehot_target('BaTiO3').reshape(1, 40, 115).shape)

# print(step_to_one_hot([2]))
# print(one_hot_to_step([(0., 0., 0., 0., 1.), (0., 1., 0., 0., 0.)]))
