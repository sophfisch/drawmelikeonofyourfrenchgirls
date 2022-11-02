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


