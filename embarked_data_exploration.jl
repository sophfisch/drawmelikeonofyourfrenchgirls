
using DataFrames
path="train.csv"
using CSV
df=CSV.read(path, DataFrame)
summary(df)
using Plots

first(df, 5)
embarked=df[!, "Embarked"]
survived=df[!, "Survived"]
#making the data integer encoded

emb=unique(df.Embarked)
pclass=df[!, "Pclass"]



#replacing missing values
Missings.replace(embarked, rand(embarked))

emb_dic=Dict(v=> k for (k,v) in enumerate(emb))

emb_num= Int[]
for h in df[!, :"Embarked"]
    push!(emb_num, emb_dic[h])
end
emb_num

using Plots

histogram(emb_num, group=survived)
##from one destination all survived
histogram(emb_num, group=pclass)

#want to see if there is a correlation between class and embarked with pearson chisq
Plots.scatter(emb_num, pclass, jitter=1)

using HypothesisTests

#High degree: If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation. Moderate degree: If the value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation. Low degree: When the value lies below + . 29, then it is said to be a small correlation.
cor(pclass, emb_num)
##no correlation between embarked and class!
using StatsPlots
groupedhist(emb_num, group = df.Survived, bar_position = :dodge)
groupedhist(emb_num, group = pclass, bar_position = :dodge)