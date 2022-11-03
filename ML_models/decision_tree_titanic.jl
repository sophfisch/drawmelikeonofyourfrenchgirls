using ScikitLearn
using DataFrames
import ScikitLearn: fit!, predict
import ScikitLearn: plot_tree
@sk_import tree: DecisionTreeClassifier
@sk_import tree: plot_tree
@sk_import tree: export_graphviz

using GraphViz
import GraphViz: Source

using CSV
X_test = CSV.read("../features/titanic_test.csv", DataFrame)

include("../features/feature_formatting_titanic.jl")

tit

X_train = Matrix(tit[:, Not(:Survived)])
Y_train = tit.Survived

X_test = Matrix(X_test)

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
clf.score(X_train, Y_train)

clf.feature_importances_

# Plot the tree
using PyCall
@sk_import tree: export_graphviz
export_graphviz(clf, out_file="mytree", class_names=["Survived", "Died"], feature_names=names(tit[:, Not(:Survived)]), leaves_parallel=true, impurity=false, rounded=true, filled=true, label="root", proportion=true)


# tune parameters

model = DecisionTreeClassifier()
nsplits = 5
parameters = Dict("max_depth" => 3:50, "min_samples_leaf" => [1, 5, 10, 25, 50, 70], "min_samples_split" => [2, 4, 10, 12, 16, 18, 25, 35])
kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
gridsearch = GridSearchCV(model, param_grid = parameters, scoring="accuracy", cv=kf, n_jobs=1)

gridsearch.fit(X_train, Y_train)
gridsearch.best_estimator_
gridsearch.best_score_

gridsearch.score(X_train, Y_train)

best_model = DecisionTreeClassifier(max_depth = 7, min_samples_split = 18, min_samples_leaf = 5)
best_model.fit(X_train, Y_train)
best_model.score(X_train, Y_train)




Y_pred = best_model.predict(X_test)

original = CSV.read("../../data/test.csv", DataFrame)

submission = DataFrame()
submission.PassengerId = original.PassengerId
submission.Survived = Y_pred
submission

CSV.write("new_submission_decision_tree.csv", submission)

plot_tree(clf, filled = true)

