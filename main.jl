using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
mach5=fit_and_evaluate(t2,tv2,te2,tev2)

k,j,h,g = data_split(data[1:5000,:],y, 1:4000, 4001:5000, shuffle =true)
mach5=fit_and_evaluate(k,j,h,g )
