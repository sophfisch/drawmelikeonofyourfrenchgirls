using DataFrames
using CSV
using VegaLite

titanic = CSV.read("train.csv", DataFrame)

#missing values?
ismissing(titanic.Sex)
ismissing(titanic.SibSp)
ismissing(titanic.Parch)

#no missing values

#sex binary encoded
sex = titanic.Sex

sex_bin = []
fem = 0
for (i,s) in enumerate(sex)
    n = 0
    if s == "female"
        n = 1
        fem += 1
    end
    push!(sex_bin, n)
end
#female = 1, male = 0
sex_df = DataFrame(sex_bin = sex_bin, survival = titanic.Survived)

sex_grouped = groupby(sex_df, :sex_bin)
sex_surv =  combine(sex_grouped, :survival => x -> sum(x) / length(x))

plot_sex = @vlplot(
    data = sex_surv,
    mark = {:point, filled = true},
    x = {:sex_bin},
    y = {:survival_function}
)


#siblings and parents as one feature -> total number of relatives on board
#it makes sennse to combine the two to get family size and not look at them individualy
#+1 for family size to count the person itself

siblings = titanic.SibSp
parents = titanic.Parch

relatives = DataFrame(relatives = parents .+ siblings, survival = titanic.Survived)

rel_grouped = groupby(relatives, :relatives)
survival_grouped = combine(rel_grouped, :survival => x -> sum(x) / length(x))

plot_relatives = @vlplot(
    data = survival_grouped,
    mark = {:point, filled = true},
    x = {:relatives},
    y = {:survival_function}
)

#age 
#replace missing values by random numnbers and group_age

using DataFrames
using CSV
using Statistics: mean, std
using Random
using Impute

age = titanic.Age
df_age = DataFrame(age = age, survival = titanic.Survived)
a = dropmissing(df_age)

#age_nomissing = transform(df_age, names(df_age) .=> Impute.locf, renamecols=false)

function replace_missing_age(df)
    new_age = df
    a = dropmissing(df)
    m = mean(a[:, 1])
    sd = std(a[:, 1])
    is_missing = ismissing.(df.age)
    for (i,v) in enumerate(is_missing)
        if v == 1
        new_age[1,i] = rand((m - sd):(m + sd))
        end
    end
    
    #dataset["Age"] = age_slice
    #dataset["Age"] = train_df["Age"].astype(int)
    #train_df["Age"].isnull().sum()
    return new_age
end

nm_age = replace_missing_age(df_age)

"""
1 -> age <= 12
2 -> 13 < age <= 18
3 -> 19 < age <= 30
4 -> 31 < age <= 45
5 -> 46 < age <= 60
6 -> 61 < age
"""

function group_age(df)
    for (i,v) in enumerate(df.age)
        #1
        if v <= 12
            df.age[i] = 1
        end
        #2
        if 12 < v <= 18
            df.age[i] = 2
        end
        #3
        if 18 < v <= 30
            df.age[i] = 3
        end
        #4
        if 30 < v <= 45
            df.age[i] = 4
        end
        #5
        if 45 < v <= 60
            df.age[i] = 5
        end
        #6
        if 60 < v
            df.age[i] = 6
        end
    end
    return 
end

group_age(nm_age)

age_grouped = groupby(nm_age, :age)
age_grouped_surv = combine(age_grouped, :survival => x -> sum(x) / length(x))

plot_age = @vlplot(
    data = age_grouped_surv,
    mark = {:point, filled = true},
    x = {:age},
    y = {:survival_function}
)



