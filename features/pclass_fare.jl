using Plots
using Statistics
using CSV
using DataFrames
using StatsPlots
using Pipe

f = raw"final project\data\train.csv"

data = CSV.read(f, DataFrame)

# missing data?

unique(ismissing.(data.Pclass)) #no missing values
unique(ismissing.(data.Fare))

count(iszero.(data.Fare)) #15 places where fare == 0
subset(data, :Fare => x -> x .== 0).Pclass

# replace Fare == 0 with mean of respective Pclass
Fare = data.Fare
mean1 = subset(data, :Pclass => x -> x .== 1).Fare |> mean
mean2 = subset(data, :Pclass => x -> x .== 2).Fare |> mean
mean3 = subset(data, :Pclass => x -> x .== 3).Fare |> mean

means = [mean1, mean2, mean3]

for i in 1:size(data, 1)
    if data.Fare[i] == 0
        Fare[i] = means[data.Pclass[i]]
    end
end

count(iszero.(Fare))
Fare
Pclass = data.Pclass


#plots

histogram(data.Fare)

boxplot(data.Survived, data.Fare, outliers = false)  # one big outlier
groupedhist(data.Pclass, group = data.Survived, bar_position = :dodge)
groupedhist(data.Fare, group = data.Pclass, bar_position = :stack, xlims = (0, 300)) #without outliers

boxplot(data.Pclass, data.Fare)


count(x -> x == 0, subset(data, :Pclass => x -> x .== 1).Survived)