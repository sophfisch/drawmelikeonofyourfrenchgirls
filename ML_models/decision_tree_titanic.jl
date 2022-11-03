using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import tree: DecisionTreeClassifier


include("features/feature_formatting_titanic.jl")

tit

X = tit[:, Not(:Survived)]
Y = tit.Survived

X.Fare = round.(Int, X.Fare)
X.Age = round.(Int, X.Age)

clf = DecisionTreeClassifier()

clf.fit(X, Y)
