using ScikitLearn
import ScikitLearn: fit!, predict
import ScikitLearn: plot_tree
@sk_import tree: DecisionTreeClassifier
@sk_import tree: plot_tree
@sk_import tree: export_graphviz

using GraphViz
import GraphViz: Source

using CSV
X_test = CSV.read("features/titanic_test.csv", DataFrame)

include("features/feature_formatting_titanic.jl")

tit

X_train = Matrix(tit[:, Not(:Survived)])
Y_train = tit.Survived

X_test = Matrix(X_test)

clf = DecisionTreeClassifier()


clf.score(X_train, Y_train)

Y_pred = clf.predict(X_test)

original = CSV.read("../data/test.csv", DataFrame)

submission = DataFrame()
submission.PassengerId = original.PassengerId
submission.Survived = Y_pred
submission

CSV.write("submission_decision_tree.csv", submission)

#= plot_tree(clf, filled = true)

dot_data = export_graphviz(clf) 
graph = GraphViz.Source(dot_data) 
graph.render("iris")  =#
clf.fit(X, Y)
