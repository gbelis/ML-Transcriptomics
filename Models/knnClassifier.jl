using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")


#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")
data3 = load_data("./data/data3.csv")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

#define KNNClassifier

using Distances, Distributions
import Pkg; Pkg.add("MLJModelInterface")
using MLJModelInterface
const MMI = MLJModelInterface
MMI.@mlj_model mutable struct SimpleKNNClassifier <: MMI.Probabilistic
    K::Int = 5 :: (_ > 0)
    metric::Distances.Metric = Euclidean()
end
function MMI.fit(::SimpleKNNClassifier, verbosity, X, y)
    fitresult = (; X = MMI.matrix(X, transpose = true), y)
    fitresult, nothing, nothing
end
function MMI.predict(model::SimpleKNNClassifier, fitresult, Xnew)
    similarities = pairwise(model.metric, fitresult.X, MMI.matrix(Xnew, transpose = true))
    [Distributions.fit(UnivariateFinite, fitresult.y[partialsortperm(col, 1:model.K)])
     for col in eachcol(similarities)]
end
function MMI.predict_mode(model::SimpleKNNClassifier, fitresult, Xnew)
    mode.(predict(model, fitresult, Xnew))
end

t2,tv2,te2,tev2 = data_split(data3[1:5000,:],y, 1:4000, 4001:5000, shuffle =true)
m = machine(SimpleKNNClassifier(K = 3), t2, tv2)
fit!(m)
predict_mode(m)
evaluate(m,t2,tv2,te2,tev2)