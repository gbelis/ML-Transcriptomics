using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
include("../data_processing.jl")
include("../models.jl")


#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=true, from_index=true)

Random.seed!(0)

###################################################### Tuning


train_data, validation_train, test_data, validation_test = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
# builder = MLJFlux.Short(n_hidden = 100,dropout = 0, σ = relu)
# mach = machine(NeuralNetworkClassifier(builder = builder,
#                 batch_size = 256,
#                 epochs = 10^4),
#                 train_data, validation_train) |> fit!

mnist_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(1500, 100, relu),
                                                          Dense(100, 3))),
                         batch_size = 256,
                         epochs = 20),
                         train_data, validation_train)
fit!(mnist_mach, verbosity = 2)
m = mean(predict_mode(mach, test_data) .== validation_test)