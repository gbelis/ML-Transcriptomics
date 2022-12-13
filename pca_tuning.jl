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


results = DataFrame(pca= 0, accuracy = 0.89)
n_folds=5

pca_dimension = collect(100:100:10000)

for i in (pca_dimension)
    m = 0.0
    data = vcat(x_train,x_test)
    
    for j in (1:n_folds)

        MLJ.transform(fit!(machine(PCA(maxoutdim = pca_dimension), data)), data)
        training_data, validation_data, test_data,validation_test = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),test_data,validation_test) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)
CSV.write("./data/pca_results.csv",results)