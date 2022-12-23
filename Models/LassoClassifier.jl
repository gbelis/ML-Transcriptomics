# Fitting and Tuning of a Lasso Classifier model

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

###################################################### Tuning
"""
Lasso Classification using cross-validation. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data
seed {int} -- value of the seed to fix
goal {int} -- number of different lambda to try
lower {float} -- value of the smallest lambda to try
upper {float} -- value of the biggest lambda to try

Hyperparameter : lambda, search between 1e-2 and 1e-8

"""

seed, goal, lower, upper = 0,8, 1e-8, 1e-2

Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)

mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                measure = MisclassificationRate()),
                                x_train, y) |>fit!
fitted_params(mach_lasso).best_model
report(mach_lasso)

# We search precisely the best lambda
seed, goal, lower, upper = 0,10, 1e-5, 1e-3

Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)
mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                measure = MisclassificationRate()),
                                x_train, y) |>fit!
fitted_params(mach_lasso).best_model
report(mach_lasso)

# And more precisely
seed, goal, lower, upper = 0,9, 6e-5, 1e-4

Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)
mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper),
                                measure = MisclassificationRate()),
                                x_train, y) |>fit!
fitted_params(mach_lasso).best_model
report(mach_lasso)
pred = predict_mode(mach_lasso, x_test)
kaggle_submit(pred, "Best_lasso_regress")


###################################################### Best Model

mach = machine(MultinomialClassifier(penalty = :l1, lambda = 8.5e-5), x_train, y) |> fit!
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "Best_lasso_regress")