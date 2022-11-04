using DataFrames
using CSV
using ScikitLearn
@sk_import ensemble: RandomForestClassifier

tit_train = CSV.read("titanic_train.csv", DataFrame)
tit_test = CSV.read("titanic_test.csv", DataFrame)


Y_train = tit_train[:,1]
X_train = Matrix(select(tit_train, Not(:Survived)))

X_test =  Matrix(tit_test)

us = CSV.read("us.csv", DataFrame)
us_m = Matrix(us)

random_forest = RandomForestClassifier(n_estimators=800)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

test = CSV.read("test.csv", DataFrame)
df_pred = DataFrame(PassengerId = test.PassengerId, Survived = Y_prediction)

CSV.write("predictions.csv", df_pred)




@sk_import model_selection: StratifiedKFold
@sk_import model_selection: GridSearchCV
@sk_import model_selection: cross_val_score



model = RandomForestClassifier()
nsplits = 5
parameters = Dict("n_estimators" => 10:10:1000)
kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
gridsearch = GridSearchCV(model, parameters, scoring="accuracy", cv=kf, n_jobs=2, verbose=0)

# train the model
gridsearch.fit(X_train,Y_train)
best_estimator = gridsearch.best_estimator_



#hyperparameter tuning
param =  Dict("max_depth" => 3:5:100, "n_estimators" => 10:10:1000)
kf = StratifiedKFold(n_splits=5, shuffle=true)
gridsearch = GridSearchCV(model, param, scoring="accuracy", cv=kf, n_jobs=2, verbose=0)

gridsearch.fit(X_train,Y_train)
best_estimator = gridsearch.best_estimator_

random_forest_md = RandomForestClassifier(max_depth=13, n_estimators=800)
random_forest_md.fit(X_train, Y_train)

Y_prediction_md = random_forest_md.predict(X_test)
pre_us = random_forest_md.predict(us_m)

random_forest_md.score(X_train, Y_train)

df_pred_md = DataFrame(PassengerId = test.PassengerId, Survived = Y_prediction_md)

CSV.write("predictions_md.csv", df_pred_md)


CSV.write("predictions_web.csv", df_pred_web)

