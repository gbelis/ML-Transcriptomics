# Best result using a linear method, a Lasso Classifier

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

#fix seed
Random.seed!(0)

model = LogisticClassifier(penalty = :l1, lambda = 8.5e-5)
mach_lasso = fit!(machine(model,x_train, y), verbosity = 1)
pred = predict_mode(mach_lasso, x_test)
kaggle_submit(pred, "LassoClassifier_best_model")