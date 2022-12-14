using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")


#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=true, from_index=true)
x_train = correlation_labels(x_train, 8000)
#mach = machine(MultinomialClassifier(penalty = :l1, lambda = 7.83e-5), x_train, y) |> fit!

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

"""

seed, goal, lower, upper = 0,5,6e-5,8e-4

Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)
mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper),
                                measure = MisclassificationRate()),
                                x_train, y) |>fit!
pred = predict_mode(mach_lasso, x_test)
kaggle_submit(pred, "LassoClassifier")

###################################################### Best Model

mach = machine(MultinomialClassifier(penalty = :l1, lambda = 7.83e-5), x_train, y) |> fit!
pred = predict_mode(mach, x_tests)
kaggle_submit(pred, "LassoClassifier_best_lambda")