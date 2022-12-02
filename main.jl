using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=true, from_index=true)

# CSV.write("./data/x_train.csv.gz", x_train)
# CSV.write("./data/x_test.csv.gz", x_test)
# CSV.write("./data/y.csv.gz", y)

#nb =pca_cumvar_plot(x_train)

data = pca(vcat(x_train,x_test),4508)
mach= multinom_class(data[1:5000,:], data[5001:end,:],y, pena= :l1, lambda =7.83e-5)


# mach1 = lasso_classifier(x_train[1:100,1:100], x_test[1:100,1:100], y[1:100], seed=0, goal=10, lower=4e-5, upper=3e-4)

# MLJ.report(mach1).best_model.lambda
# plot(mach1)
# confusion_matrix(predict_mode(mach1),y )
