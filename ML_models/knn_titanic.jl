using ScikitLearn
import ScikitLearn: fit!, predict
using LsqFit
@sk_import neighbors: NearestNeighbors


@sk_import neighbors: KNeighborsClassifier

import ScikitLearn: CrossValidation
@sk_import model_selection: StratifiedKFold
@sk_import model_selection: GridSearchCV
@sk_import metrics: accuracy_score

#use titanic_train.csv as df


y=select(df, :Survived)
y=convert(Array, y)
X=select(df, Not(:Survived))
X=Matrix(X)
# KNN
function KNN_classification(X_train, y_train; nsplits=5, scoring="accuracy", n_jobs=1)
    model2 = KNeighborsClassifier()
    parameters = Dict("n_neighbors" => 16:2:18, "weights" => ("uniform", "distance"), "algorithm"=>("auto", "ball_tree", "kd_tree", "brute"))
    kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
    gridsearch = GridSearchCV(model2, parameters, scoring=scoring, cv=kf, n_jobs=n_jobs, verbose=0)
    # train the model
    fit!(gridsearch,X_train,y_train)
    best_estimator = gridsearch.best_estimator_
    return best_estimator
end

best_est2 = KNN_classification(X,y)
model = KNeighborsClassifier(best_est2)



