
using CSV
using Plots
using Statistics
using CSV
using DataFrames
using StatsPlots
using Pipe



# GETTING ALL FEATURES READY
############## EMBARKED ################
path="train.csv"

df=CSV.read(path, DataFrame)

f = "test.csv"

df_test = CSV.read(f, DataFrame)

function embarked(df)
    embark=df[!, "Embarked"]
    #making the data integer encoded
    emb=unique(df.Embarked)
    #replacing missing values
    Missings.replace(embark, rand(embark))
    emb_dic=Dict(v=> k for (k,v) in enumerate(emb))
    emb_num= Int[]
    for h in df[!, :"Embarked"]
        push!(emb_num, emb_dic[h])
    end
    emb=emb_num
    return emb
end

emb=embarked(df)
emb_test=embarked(df_test)

################ FARE ##################################

# replace Fare == 0 with mean of respective Pclass
function fare(df)
    Fare = df.Fare
    mean1 = subset(dropmissing(df), :Pclass => x -> x .== 1).Fare |> mean
    mean2 = subset(dropmissing(df), :Pclass => x -> x .== 2).Fare |> mean
    mean3 = subset(dropmissing(df), :Pclass => x -> x .== 3).Fare |> mean

    means = [mean1, mean2, mean3]
    mis = ismissing.(Fare)

    for i in 1:size(df, 1)
        if mis[i] == 0 
            if df.Fare[i] == 0
                Fare[i] = means[df.Pclass[i]]
            end
        end
    end

    for i in 1:length(mis)
        if mis[i] == 1
            Fare[i] = means[df.Pclass[i]]
        end
    end
    return Fare
end

fare_titanic = fare(df)
fare_titanic_test = fare(df_test)

############################ AGE ########################

function replace_missing_age(df)
    new_age = df.Age
    a = dropmissing(df)
    m = mean(a.Age)
    sd = std(a.Age)
    is_missing = ismissing.(df.Age)
    for (i,v) in enumerate(is_missing)
        if v == 1
        new_age[i] = rand((m - sd):(m + sd))
        end
    end
    
    #dataset["Age"] = age_slice
    #dataset["Age"] = train_df["Age"].astype(int)
    #train_df["Age"].isnull().sum()
    return new_age
end

nm_age = replace_missing_age(df)
nm_age_test = replace_missing_age(df_test)


function group_age(df)
    age=replace_missing_age(df)
    for (i,v) in enumerate(age)
        #1
        if v <= 12
            age[i] = 1
        end
        #2
        if 12 < v <= 18
            age[i] = 2
        end
        #3
        if 18 < v <= 30
            age[i] = 3
        end
        #4
        if 30 < v <= 45
            age[i] = 4
        end
        #5
        if 45 < v <= 60
            age[i] = 5
        end
        #6
        if 60 < v
            age[i] = 6
        end
    end
    return age
end
age=group_age(df)
age_test=group_age(df_test)

######################## SEX ###################

function sex_binary(df)
    sex_bin = []
    for (i,s) in enumerate(df.Sex)
        n = 0
        if s == "female"
        n = 1
        end
        push!(sex_bin, n)
    end
    return sex_bin
end
sex=sex_binary(df)
sex_test=sex_binary(df_test)

######################## SIBLINGS RELATIVES ####################

siblings = df.SibSp
siblings_test = df_test.SibSp

parents = df.Parch
parents_test = df_test.Parch

relatives = parents .+ siblings
relatives_test = parents_test .+ siblings_test


tit = DataFrame(Survived=df.Survived, Pclass=df.Pclass, Sex=sex, Age=age, Relatives=relatives, Fare=fare_titanic, Embarked=emb) 
tit_test = DataFrame(Pclass=df_test.Pclass, Sex=sex_test, Age=age_test, Relatives=relatives_test, Fare=fare_titanic_test, Embarked=emb_test) 

using ScikitLearn
import ScikitLearn: fit!, predict
@sk_import linear_model: LogisticRegression
@sk_import metrics: f1_score

X_train = zeros(size(tit,1), size(tit, 2)-1)
for i in 1:size(X_train, 2)
    X_train[:, i] = tit[!, i+1]
end

X_test = zeros(size(tit_test,1), size(tit_test, 2))
for i in 1:size(X_test, 2)
    X_test[:, i] = tit_test[!, i]
end

Y_train = tit[!, :Survived]

model = LogisticRegression()
logistic = fit!(model, X_train, Y_train)
ypred = logistic.predict(X_train)
coef = logistic.intercept_, logistic.coef_

logistic.score(X_train, Y_train)

ypred_test = logistic.predict(X_test)

logreg_titanic = DataFrame(PassengerId = df_test.PassengerId, Survived = ypred_test)
CSV.write("logreg_titanic.csv", logreg_titanic)

#acc_log = round(logistic.score(X_train, Y_train) * 100, 2)