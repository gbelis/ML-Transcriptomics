using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, MLJClusteringInterface, MLJMultivariateStatsInterface
include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

mach= multinom_class(x_train,x_test,y, pena= :l2, lambda =1e-2)

mach1 = lasso_classifier(x_train[1:100,1:100], x_test[1:100,1:100], y[1:100], seed=0, goal=10, lower=4e-5, upper=3e-4)

MLJ.report(mach1).best_model.lambda
plot(mach1)
confusion_matrix(predict_mode(mach1),y )
