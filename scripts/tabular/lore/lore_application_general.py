from lore.lorem import LOREM
from lore.datamanager import *
from lore.util import record2str, mixed_distance_idx, neuclidean, multilabel2str, nmeandev
from scipy.spatial.distance import jaccard
import pickle
import pandas as pd

############
######
### load LORE at: https://github.com/rinziv/LORE_ext
######
############

def main():

    dataset = 'adult'
    model = 'catboost'
    types = 'rndgen'
    title = "../../../../tabular/datasets/test_set_" + dataset + "_strat.p"
    test = open(title, "rb")
    test_set = pickle.load(test)
    pickle_in = open("../../../../tabular/results/trained_" + model + "_" + dataset + ".p", "rb")
    bb = pickle.load(pickle_in)


    class_name = 'class'
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(test_set, class_name)
    if features_map:
        features_map_inv = dict()
        for idx, idx_dict in features_map.items():
            for k, v in idx_dict.items():
                features_map_inv[v] = idx

    unadmittible_features = {'age': None, 'race': None, 'sex': None,
                             'native-country': None, 'marital-status': None, }

    def bb_predict(X):
        return bb.predict(X)


    test_val = test_set.values

    explainer = LOREM(test_set, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type=types, categorical_use_prob=True,
                      continuous_fun_estimation=True, size=1000, ocr=0.1, multi_label=False, one_vs_rest=False,
                      random_state=0, verbose=False, ngen=10)


    explainer.set_unfeasibible_features(unadmittible_features)
    title = 'lore_' + model + '_' + dataset + '_' + types +'.p'
    explanations = list()
    with open(title, "ab") as pickle_file:
        for i2e, x in enumerate(test_val):
            exp = explainer.explain_instance(x, samples=1000, use_weights=True, metric=neuclidean)
            explanations.append((x, exp))
            pickle.dump(explanations, pickle_file)


if __name__ == "__main__":
    main()

