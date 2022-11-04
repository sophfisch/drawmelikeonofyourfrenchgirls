#seing which predictions would be best_est2
using StatsBase
using CSV
using DataFrames
knn=CSV.read("submission_knn", DataFrame)
tree=CSV.read("new_submission_decision_tree.csv", DataFrame)
forest=CSV.read("predictions_web.csv", DataFrame)


k=knn.Survived
t=tree.Survived
f=forest.Survived

preds=hcat(k,t,f)
preds[3, :]
for i in 1:size(preds, 1)
    a=countmap(preds[i, :])
    print(a)
end

function countmemb1(y)
    arr=zeros(size(y, 1))
    for i in 1:size(y, 1)
        print(sum(y[i, :]))
        if sum(y[i, :]) <= 1
            arr[i]=0
        else 
            arr[i]=1
        end
    end
    return convert.(Int, arr)
end

ar=countmemb1(preds)

print(ar)


q="test.csv"
org=CSV.read(q, DataFrame)
submission = DataFrame()
submission.PassengerId = org.PassengerId
submission.Survived = ar
submission


CSV.write("final_submission.csv", submission)
first(df)