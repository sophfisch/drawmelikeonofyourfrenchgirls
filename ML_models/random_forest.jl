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



model = RandomForestClassifier()
nsplits = 5
parameters = Dict("n_estimators" => 10:10:1000)
kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
gridsearch = GridSearchCV(model, parameters, scoring="accuracy", cv=kf, n_jobs=2, verbose=0)

# train the model
fit!(gridsearch,X_train,Y_train)
best_estimator = gridsearch.best_estimator_



#hyperparameter tuning
param =  Dict("max_depth" => 3:100, "n_estimators" => 10:10:1000)
kf = StratifiedKFold(n_splits=nsplits, shuffle=true)
gridsearch = GridSearchCV(model, param, scoring="accuracy", cv=kf, n_jobs=2, verbose=0)

fit!(gridsearch,X_train,Y_train)
best_estimator = gridsearch.best_estimator_

random_forest_md = RandomForestClassifier(n_estimators=800)
random_forest_md.fit(X_train, Y_train)

Y_prediction_md = random_forest.predict(X_test)

random_forest_md.score(X_train, Y_train)



rf = RandomForestClassifier(n_estimators=100, max_features="auto", oob_score=true, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param, n_jobs=-1)
clf.fit(X_train, Y_train)
clf.best_estimator_



random_forest_web = RandomForestClassifier(n_estimators = 100, max_features = "auto", criterion="gini", min_samples_split=10, min_samples_leaf = 1, n_jobs=-1, oob_score=true, random_state=1)
random_forest_web.fit(X_train, Y_train)

Y_prediction_web = random_forest_web.predict(X_test)

random_forest_web.score(X_train, Y_train)


#feature importance
importances = DataFrame(feature = ("Survived","Pclass","Sex","Age","Relatives","Fare","Embarked"), importance = random_forest.feature_importances_)
importances = importances.sort_values("importance",ascending=false).set_index("feature")
importances.head(15)