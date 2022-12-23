# Best Short Neural Network

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#standardize data
x_train, x_test = norm(x_train, x_test)

#select 6000 best genes according to the mean difference
mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))
mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))
mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))

results_mean= DataFrame(gene = names(x_train), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, diff1=abs.(mean_CBP-mean_eGFP), diff2=abs.(mean_eGFP -mean_KAT5), diff3=(abs.(mean_CBP -mean_KAT5)))

sort!(results_mean, [:diff1], rev=true)
selection1 = results_mean[1:6000,:] 
sort!(results_mean, [:diff2], rev=true)
selection2 = results_mean[1:6000,:]
sort!(results_mean, [:diff3], rev=true)
selection3 = results_mean[1:6000,:]

x_train2 = select(x_train, unique([selection1.gene; selection2.gene; selection3.gene]))

data = vcat(x_train2,x_test)

# Do a PCA to reduce the features to 3000
data = MLJ.transform(fit!(machine(PCA(maxoutdim = 3000), data)), data)

x_train2= data[1:5000,:]
x_test = data[5001:8093,:]

#fix seed
Random.seed!(0)

#Neural network fitting with best hyperparameters found in NN.jl
model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout = 0.5),
                optimiser = ADAM(),
                batch_size = 128,
                epochs = 2000,
                alpha = 0.25)

mach = fit!(machine(model,x_train2, y), verbosity = 1)

pred = predict_mode(mach, x_test)
kaggle_submit(pred, "Submission_NN_mean")