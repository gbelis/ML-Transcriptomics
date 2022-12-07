using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("./data_processing.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
pred = predict_mode(mach, x_tests)
kaggle_submit(pred, "MultinomialClassifier")