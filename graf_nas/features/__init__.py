from graf_nas.features.nasbench101 import feature_func_dict as nb101_dict
from graf_nas.features.nasbench201 import feature_func_dict as nb201_dict
from graf_nas.features.tnb101_macro import feature_func_dict as tnb101_macro_dict
from graf_nas.features.darts import feature_func_dict as darts_dict


feature_dicts = {
    'nb101': nb101_dict,
    'nb201': nb201_dict,
    'tnb101_micro': nb201_dict,
    'tnb101_macro': tnb101_macro_dict,
    'darts': darts_dict
}
