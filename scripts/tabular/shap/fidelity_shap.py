import pickle
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

#to evaluate the fidelity we define and fit a simple model in which the output class depends on the shap values

#retrieve shap values
model = 'xgb'
dataset = 'adult'
filename = 'shap_values_'+model+'_'+dataset+'.p'
shap_values = list()
max_n = 3000
count =0
with (open(filename, "rb")) as openfile:
    while True:
        if count < max_n:
            try:
                shap_values.append(pickle.load(openfile))
                count += 1
            except EOFError:
                break
        else:
            break

shaps = shap_values[max_n-1]
sums = []
for s in shaps:
    sums.append(sum(s))

#define the new labels
results = []
for s in sums:
    if s > 0:
        results.append(1)
    else:
        results.append(0)

#retrieve the test set
title = "../../tabular/datasets/test_set_"+dataset+"_strat.p"
test = open(title,"rb")
test_set = pickle.load(test)
tests = test_set.head(len(shaps))

#define and fit the simple model
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
clf_tree.fit(tests.values, results)
labels = clf_tree.predict(tests.values)
print(classification_report(results, labels))
