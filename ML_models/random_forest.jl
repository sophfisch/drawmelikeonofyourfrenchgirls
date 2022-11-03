using DataFrames
using CSV
using ScikitLearn
@sk_import ensemble: RandomForestClassifier

tit_train = CSV.read("titanic_train.csv", DataFrame)
tit_test = CSV.read("titanic_test.csv", DataFrame)

Y_train = tit_train[:,1]
X_train = Matrix(select(tit_train, Not(:Survived)))

X_test =  Matrix(tit_test)

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
@sk_import ensemble: oob_score_


model = RandomForestClassifier()
nsplits = 5
parameters = Dict("n_estimators" => 10:10:1000)
kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
gridsearch = GridSearchCV(model, parameters, scoring="accuracy", cv=kf, n_jobs=2, verbose=0)

# train the model
fit!(gridsearch,X_train,Y_train)
best_estimator = gridsearch.best_estimator_

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")

#hyperparameter tuning
param =  Dict("criterion" => ["gini", "entropy"], "min_samples_leaf" => [1, 5, 10, 25, 50, 70], "min_samples_split" => [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators" => [100, 400, 700, 1000, 1500])

rf = RandomForestClassifier(n_estimators=100, max_features="auto", oob_score=true, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param, n_jobs=-1)
clf.fit(X_train, Y_train)
clf.best_estimator_


#RandomForestClassifier(criterion='entropy', max_features='auto', min_samples_split=10, n_jobs=-1, oob_score=True, random_state=1)
random_forest_est = RandomForestClassifier(criterion="entropy", max_features="auto", min_samples_split=10, n_jobs=-1, oob_score=true, random_state=1)
random_forest_est.fit(X_train, Y_train)

Y_prediction_est = random_forest_est.predict(X_test)

random_forest_est.score(X_train, Y_train)



random_forest_web = RandomForestClassifier(n_estimators = 100, max_features = "auto", criterion="gini", min_samples_split=10, min_samples_leaf = 1, n_jobs=-1, oob_score=true, random_state=1)
random_forest_web.fit(X_train, Y_train)

Y_prediction_web = random_forest_web.predict(X_test)

random_forest_web.score(X_train, Y_train)


