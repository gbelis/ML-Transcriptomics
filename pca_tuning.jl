using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, MLJFlux, Flux
import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS
include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
train_df = nothing
test_df = nothing
x_test = nothing

#normalisation
x_train, x_test = norm(x_train, x_test)
x_train = MLJ.transform(fit!(machine(Standardizer(), x_train)), x_train)

x_train = load_data("./data/x_train_norm.csv.gz")

################################## PCA
results = DataFrame([[],[]], ["pca","accuracy"])
n_folds=5

# Tuning of pca : hyperparameters maxoutdim, search space : [100,10000]
pca_dimension = collect(100:100:10000)

for i in (pca_dimension)
    m = 0.0
    data = vcat(x_train,x_test)
    
    for j in (1:n_folds)
        #cross validation
        data = MLJ.transform(fit!(machine(PCA(maxoutdim = pca_dimension), data)), data)
        training_data, validation_data, test_data,validation_test = data_split(data[1:5000,:],y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),training_data,validation_data) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

#save results
CSV.write("./data/pca_results_pca.csv",results)

######################################## PCA Visualization

#Visualize PCA with px.scatter_3d, we  can visualize the 3 first dimensionsof a pca

#fit a PCA
data= vcat(x_train,x_test)
mach_pca = fit!(machine(PCA(maxoutdim = 3), data))
data_pca = MLJ.transform(mach_pca,data)[1:5000,:]
#compute the variance
explained_variance = report(mach_pca).principalvars
explained_variance ./= sum(explained_variance)
explained_variance .*= 100
dimensions = Symbol.(names(data_pca))
data_pca.y=y
total_var = report(mach_pca).tprincipalvar / report(mach_pca).tvar

#plot
p= PlotlyJS.plot(data_pca, x=:x1, y=:x2, z=:x3, color=:y, kind="scatter3d", mode="markers" ,labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...), 
Layout(title="PCA of the 3 most relevant components. \n Total explained variance: $(round(total_var, digits=2),
scene = attr(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="Z AXIS TITLE"))"))

#save as a html file
open("./pca_plot.html", "w") do io PlotlyBase.to_html(io, p.plot) end

######################################## PCA Variance plot
#Plotting explained variance, to see how much variance PCA is able to explain 
# as we you increase the number of components, in order to decide how many dimensions to ultimately keep or analyze

pca_gene = fit!(machine(PCA(), x_train), verbosity = 0);
vars = report(pca_gene).principalvars ./ report(pca_gene).tvar
p1 = plot(vars, label = nothing, yscale = :log10,
            xlabel = "component", ylabel = "proportion of variance explained")
p2 = plot(cumsum(vars),
            label = nothing, xlabel = "component",
            ylabel = "cumulative prop. of variance explained")
p3 = plot(p1, p2, layout = (1, 2), size = (700, 400))
report(pca_gene).principalvars

Plots.savefig("variance_plot.png")

#save as a html file
open(".pca_cumvar_plots.html", "w") do io PlotlyBase.to_html(io, p3.plot) end


################################## UMAP
# Tuning of pca : hyperparameter n_components, search space : [1000,5000]

results = DataFrame(umap_component= 0, accuracy = 0.89)
n_folds=5

umap_dimension = collect(1000:1000:5000)

for i in (umap_dimension)
    m = 0.0

    for j in (1:n_folds)
        umap_proj = umap(Array(x_train)', i, min_dist = .5, n_neighbors = 10);
        training_data, validation_data, test_data,validation_test = data_split(transpose(umap_proj),y, 1:4000, 4001:5000, shuffle =true)
        training_data2 = DataFrame(x1 =training_data[:,1], x2=training_data[:,2],x3=training_data[:,3])
        mach = machine(MultinomialClassifier(penalty =:none),training_data2,validation_data) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

######################################## UMAP Visualization

umap_proj = umap(Array(x_train)', 3 )

umap_proj.y=y
PlotlyJS.plot(umap_proj, x=:x1, y=:x2, z=:x3, color=:y, kind="scatter3d", mode="markers" ,labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...), 
Layout(title="Total explained variance: $(round(total_var, digits=2))"))

######################################## Visualization of best predictors

#select the best predictors using a t-test
pred_names = get_names_len(x_train, y, 6000)
x_train = select(x_train, pred_names)
x_test = select(x_test, names(x_train))

println(pred_names[1:10])

x_train.label=y

#correlation plot
@df x_train corrplot([:Mid1 :Polr1b :Hexb :Gm42418],
                     grid = false, fillcolor = cgrad(), size = (600, 600), 
                     title="Correlation plot of the 4 more important genes", titlelocation = :right)
Plots.savefig("corr_plot_norm.png")
x_train= x_train[:,1:24101]

#splom plot
x_train
x_train.labels = y
features = [:Mid1, :Polr1b, :Hexb, :Gm42418]
p= PlotlyJS.plot(x_train, dimensions=features, color= :labels, kind="splom", Layout(title="Plot for the 4 more important genes"))

#3d plot
x_train.y = y
p= PlotlyJS.plot(x_train, x=:Mid1, y= :Polr1b, z =:Hexb, color=:y, kind="scatter3d", mode="markers", Layout(title="PCA of the 3 most relevant components"))

#open("../Plots/pca_plot.html", "w") do io PlotlyBase.to_html(io, p.plot) end