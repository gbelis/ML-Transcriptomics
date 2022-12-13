using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS
include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

#################### PCA
results = DataFrame(pca= 0, accuracy = 0.89)
n_folds=5

pca_dimension = collect(100:100:10000)

for i in (pca_dimension)
    m = 0.0
    data = vcat(x_train,x_test)
    
    for j in (1:n_folds)

        #pas juste a refaire
        data = MLJ.transform(fit!(machine(PCA(maxoutdim = pca_dimension), data)), data)
        training_data, validation_data, test_data,validation_test = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),training_data,validation_data) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)
CSV.write("./data/pca_results.csv",results)

########################################PCA Visualization
data= vcat(x_train,x_test)
mach_pca = fit!(machine(PCA(maxoutdim = 3), data))
data_pca = MLJ.transform(mach_pca,data)[1:5000,:]
explained_variance = report(mach_pca).principalvars
explained_variance ./= sum(explained_variance)
explained_variance .*= 100
dimensions = Symbol.(names(data_pca))
data_pca.y=y

total_var = report(mach_pca).tprincipalvar / report(mach_pca).tvar
PlotlyJS.plot(data_pca, x=:x1, y=:x2, z=:x3, color=:y, kind="scatter3d", mode="markers" ,labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...), 
Layout(title="Total explained variance: $(round(total_var, digits=2))"))


function pca_cumvar_plot(training_data)
    pca_gene = fit!(machine(PCA(), training_data), verbosity = 0);
    vars = report(pca_gene).principalvars ./ report(pca_gene).tvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
    return report(pca_gene).principalvars
end


#################### UMAP

results = DataFrame(umap_component= 0, accuracy = 0.89)
n_folds=5

umap_dimension = collect(100:1000:5100)

for i in (umap_dimension)
    m = 0.0

    for j in (1:n_folds)
        umap_proj = umap(Array(x_train)', 3, min_dist = .5, n_neighbors = 10);
        training_data, validation_data, test_data,validation_test = data_split(transpose(umap_proj),y, 1:4000, 4001:5000, shuffle =true)
        training_data2 = DataFrame(x1 =training_data[:,1], x2=training_data[:,2],x3=training_data[:,3])
        mach = machine(MultinomialClassifier(penalty =:none),training_data2,validation_data) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

########################################UMAP Visualization
umap_proj.y=y
PlotlyJS.plot(umap_proj, x=:x1, y=:x2, z=:x3, color=:y, kind="scatter3d", mode="markers" ,labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...), 
Layout(title="Total explained variance: $(round(total_var, digits=2))"))