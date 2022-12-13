using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface, MLJFlux, Flux, UMAP
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")


#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

y = coerce!(train_df, :labels => Multiclass).labels
x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
x_train, x_test = norm(x_train, x_test)

