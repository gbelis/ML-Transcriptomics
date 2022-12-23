# Fitting and Tuning of a RandomForest Classifier model

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLJ, Plots, MLJFlux, Flux, OpenML, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, MLJMultivariateStatsInterface
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#standardize data
x_train, x_test = norm(x_train, x_test)

###################################################### Tuning
"""
RandomForest Classification using cross-validation. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data
seed {int} -- value of the seed to fix

Hyperparameters :n_trees between 10 and 1500, max_depth between -1 and 10

"""

seed = 0
Random.seed!(seed)
mach_forest = machine(TunedModel(model=RandomForestClassifier(),
                                        resampling = CV(nfolds=3),
                                        tuning = Grid(goal = 15),
                                        ranges = [range(RandomForestClassifier(), :n_trees, values=[10,100,500,1000,1500]),
                                                 range(RandomForestClassifier(), :max_depth, values=[-1,5,10])],
                                        measure = MisclassificationRate()),
                                        x_train, y) |>fit!

training_error = mean(predict_mode(mach_forest, x_train) .!= y)
pred = predict_mode(mach_forest, x_test)
kaggle_submit(pred, "RandomForestClassifier_norm")

fitted_params(mach_forest).best_model