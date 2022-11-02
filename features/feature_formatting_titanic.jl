
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

function embarked(df)
    embark=df[!, "Embarked"]
    survived=df[!, "Survived"]
    #making the data integer encoded
    emb=unique(df.Embarked)
    pclass=df[!, "Pclass"]
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

################ FARE ##################################

# replace Fare == 0 with mean of respective Pclass
function fare(df)
    Fare = df.Fare
    mean1 = subset(df, :Pclass => x -> x .== 1).Fare |> mean
    mean2 = subset(df, :Pclass => x -> x .== 2).Fare |> mean
    mean3 = subset(df, :Pclass => x -> x .== 3).Fare |> mean

    means = [mean1, mean2, mean3]
    mis = ismissing.(Fare)

    for i in 1:size(df, 1)
        if mis[i] == false 
            if df.Fare[i] == 0
                Fare[i] = means[df.Pclass[i]]
            end
        end
    end
    for i in 1:length(mis)
        if mis[i]
            Fare[i] = means[df.Pclass[i]]
        end
    end
    return Fare
end

Fare = fare(data)

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
######################## SIBLINGS RELATIVES ####################

siblings = df.SibSp
parents = df.Parch

relatives = parents .+ siblings



tit=DataFrame(Survived=df.Survived, Pclass=df.Pclass, Sex=sex, Age=age, Relatives=relatives, Fare=fare, Embarked=emb) 



