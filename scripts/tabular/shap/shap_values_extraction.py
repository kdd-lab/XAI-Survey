import pandas as pd
import xgboost as xgb
import pickle
import shap

shap.initjs()


# select the dataset, the model to retrieve and its train and test set
dataset = 'adult'
model = 'xgb'

title = "../../tabular/datasets/train_set_"+dataset+"_strat.p"
train = open(title,"rb")
train_set = pickle.load(train)
title = "../../tabular/datasets/train_label_"+dataset+"_strat.p"
train_l = open(title,"rb")
train_label = pickle.load(train_l)
title = "../../tabular/datasets/test_set_"+dataset+"_strat.p"
test = open(title,"rb")
test_set = pickle.load(test)
title = "../../tabular/datasets/test_label_"+dataset+"_strat.p"
test_l = open(title,"rb")
test_label = pickle.load(test_l)

feature_names = train_set.columns


# load the model
title = "../../tabular/results/trained_"+model+"_"+dataset+".p"
bb = pickle.load(open(title,"rb"))

## kernel shap sends data as numpy array which has no column names, so we fix it
def xgb_predict(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=feature_names)
    return bb.predict(data_asframe)

# Kernel SHAP
X_summary = shap.kmeans(train_set, 200)
shap_kernel_explainer = shap.KernelExplainer(xgb_predict, X_summary)


# shapely values with kernel SHAP
title = 'shap_values_'+model+'_'+dataset+'.p'
shap_values_list = list()
test_set = test_set.values
with open(title, "ab") as pickle_file:
    for t in test_set:
        shap_values = shap_kernel_explainer.shap_values(t)
        shap_values_list.append(shap_values)
        pickle.dump(shap_values_list, pickle_file)


