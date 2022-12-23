# Fitting of a simple Multinomial Classifier model

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")

"""
Multinomial Classification. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data

"""

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

#Fit a linear Logistic model
mach = machine(LogisticClassifier(penalty = :none), x_train, y) |> fit!
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "MultinomialClassifier")