using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
data = load_data("./data/PCA_4500.csv.gz")
data2 = load_data("./data/PCA_4500.csv.gz")

mach2= multinom_class(x_train,x_test,y, pena= :l2, lambda =1e-2, title ="-2")
#mach2pca= multinom_class(data[1:5000,:],data[50001:8093,:],y, pena= :l2, lambda =1e-2, title ="-2")
mach3pca = multinom_class(data[1:5000,:],data[50001:8093,:],y, pena= :l2, lambda =1e-3, title ="-3")
mach4pca = multinom_class(data[1:5000,:],data[50001:8093,:],y, pena= :l2, lambda =1e-4, title ="-4")
mach4pca = multinom_class(data[1:5000,:],data[50001:8093,:],y, pena= :l2, lambda =1e-5, title ="-5")

t,tv,te,tev = data_split(x_train[1:1000,1:1000],y[1:1000], 1:500, 501:1000, shuffle =true)
mach, error = fit_and_evaluate(t,tv,te,tev)
println(error)
mean(predict_mode(mach, te) .!= tev)

t1,tv1,te1,tev1 = data_split(data[1:1000,1:1000],y[1:1000], 1:500, 501:1000, shuffle =true)
mach1, tt=fit_and_evaluate(t1,tv1,te1,tev1)
mean(predict_mode(mach1, te1) .!= tev1)

#nb =pca_cumvar_plot(x_train)
#data = pca(vcat(x_train,x_test),4508)
# mach1 = lasso_classifier(x_train[1:100,1:100], x_test[1:100,1:100], y[1:100], seed=0, goal=10, lower=4e-5, upper=3e-4)

# MLJ.report(mach1).best_model.lambda
plot(mach1)
confusion_matrix(predict_mode(mach),y )
