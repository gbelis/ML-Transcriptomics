# Fitting and Tuning of a Ridge Classifier model

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")

"""
Ridge Classification using cross-validation. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data
seed {int} -- value of the seed to fix
goal {int} -- number of different lambda to try
lower {float} -- value of the smallest lambda to try
upper {float} -- value of the biggest lambda to try

Hyperparameter : lambda, search between 1e-2 and 1e-6

"""

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

###################################################### Tuning

seed , goal, lower, upper = 0,5,1e-6,1e-2

Random.seed!(seed)
model = LogisticClassifier(penalty = :l2)
tuned_model_ridge = TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                measure = MisclassificationRate())
mach_ridge = machine(tuned_model_ridge, x_train, y) |>fit!
pred = predict_mode(mach_ridge, x_test)
kaggle_submit(pred, "RidgeClassifier")

###################################################### Best Model

mach = machine(MultinomialClassifier(penalty = :l2, lambda = 1e-3), x_train, y) |> fit!
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "RidgeClassifierClassifier_best_hyperparameter")