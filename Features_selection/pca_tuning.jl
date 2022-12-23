# This file contains all PCA tuning .

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, MLJFlux, Flux
import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

# to use less RAM
train_df = nothing
test_df = nothing

#normalisation
x_train, x_test = norm(x_train, x_test)

################################## PCA
results = DataFrame([[],[]], ["pca","accuracy"])
n_folds=5

# Tuning of pca : hyperparameters maxoutdim, search space : [100,10000]
pca_dimension = collect(100:100:10000)

pca_dimension = 3

for i in (pca_dimension)
    m = 0.0
    data = vcat(x_train,x_test)
    
    for j in (1:n_folds)
        #cross validation
        data = MLJ.transform(fit!(machine(PCA(maxoutdim = pca_dimension), data)), data)
        training_data, validation_data, test_data,validation_test = data_split(data[1:5000,:],y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),training_data,validation_data) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)
