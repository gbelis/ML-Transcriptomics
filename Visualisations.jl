# This file contains code for all figures 

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
import Pkg; Pkg.add("PlotlyJS")
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface, MLJFlux, Flux, UMAP, PlotlyJS
include("./data_processing.jl")
include("./models.jl")


train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))
y = coerce!(train_df, :labels => Multiclass).labels


# fix seed
Random.seed!(0)
#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#normalisation
x_train_norm, x_test_norm = norm(x_train, x_test)


x_train_sel = correlation_labels(x_train_norm, 5000)
x_test_sel = select(x_test, names(x_train_sel))

data= vcat(x_train_sel,x_test_sel)
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


data_log = log2.(vcat(x_train, x_test).+1)
x_train_log, x_test_log = norm(data_log[1:5000,:], data_log[5001:end,:])
x_train_sel_log = correlation_labels(x_train_log, 5000)
x_test_sel_log = select(x_test_log, names(x_train_sel_log))

data_log = vcat(x_train_sel_log,x_test_sel_log)
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

names(x_train_sel) == names(x_train_sel_log)

########################################  
# Histogram of distribution of label in the training set

df = DataFrame( label = ["CBP","KAT5","eGFP"], 
                sample_nb = [length(x_train_norm[(y.=="CBP"),:][:,1]), 
                            length(x_train_norm[(y.=="KAT5"),:][:,1]), 
                            length(x_train_norm[(y.=="eGFP"),:][:,1])])

p1 = PlotlyJS.plot(df, x=:label, y=:sample_nb, height=300, kind="bar", Layout(title="Distributions of experimental conditions in the training data"))
open("./histogram.html", "w") do io PlotlyBase.to_html(io, p1.plot) end

######################################## 
#Confusion matrix

train_data, validation_train, test_data, validation_test = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(MultinomialClassifier(penalty =:none),train_data, validation_train) |>fit!
m = mean(predict_mode(mach, test_data) .== validation_test)
confusion_matrix(predict_mode(mach, test_data), validation_test)


#######################################
#PCA Visualization
#Visualize PCA with px.scatter_3d, we can visualize the 3 first dimensions of a pca
data = remove_prop_predictors(remove_constant_predictors(select(train_df, Not(:labels))))

data_std = MLJ.transform(fit!(machine(Standardizer(), data)))


#fit a PCA
mach_pca = fit!(machine(PCA(maxoutdim = 3), data_std))
data_pca = MLJ.transform(mach_pca,data_std)
#compute the variance
explained_variance = report(mach_pca).principalvars
explained_variance ./= sum(explained_variance)
explained_variance .*= 100
dimensions = Symbol.(names(data_pca))
data_pca.Labels=y
total_var = report(mach_pca).tprincipalvar / report(mach_pca).tvar

#plot
p= PlotlyJS.plot(data_pca, x=:x1, y=:x2, z=:x3, color=:Labels, kind="scatter3d", mode="markers" ,labels=attr(;[Symbol("x", i) => "PC $i" for i in 1:3]...), 
Layout(title = "PCA: 3 Components", scene = attr(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3")))

#save as a html file
open("./Plots/pca_plot.html", "w") do io PlotlyBase.to_html(io, p.plot) end


#######################################
#Log PCA
data = remove_prop_predictors(remove_constant_predictors(select(train_df, Not(:labels))))
data_log  = log.(data .+ 1)
data_log_std = MLJ.transform(fit!(machine(Standardizer(), data_log)))
mach_log_pca = fit!(machine(PCA(maxoutdim = 3), data_log_std))
data_log_pca = MLJ.transform(mach_log_pca,data_log_std)
data_log_pca.Labels=y

p= PlotlyJS.plot(data_log_pca, x=:x1, y=:x2, z=:x3, color=:Labels, kind="scatter3d", mode="markers", 
Layout(title = "Log PCA: 3 Components", scene = attr(xaxis_title="PCA1", yaxis_title="PCA2", zaxis_title="PCA3")))

open("./Plots/pca_plot_log.html", "w") do io PlotlyBase.to_html(io, p.plot) end

#######################################
#PCA Variance plot
#Plotting explained variance, to see how much variance PCA is able to explain 
# as we increase the number of components, in order to decide how many dimensions to ultimately keep or analyze

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

#######################################
#Visualization of best predictors

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

#save as html file
open("../Plots/pca_plot.html", "w") do io PlotlyBase.to_html(io, p.plot) end

######################################################
# Plotting 3 predicotrs time 


pred_plot_1 = MLJ.transform(fit!(machine(Standardizer(), select(train_df, ["Mid1", "Hexb", "Gm42418"]))))
pred_plot_1.labels = y
p = PlotlyJS.plot(pred_plot_1, x=:Mid1, y=:Hexb, z=:Gm42418, color=:labels, kind="scatter3d", mode="markers",  Layout(title="Mid1, Hexb, Gm42418"))
#save as html file
open("./Plots/pred_plot_Mid1_Hexb_Gm42418.html", "w") do io PlotlyBase.to_html(io, p.plot) end



pred_plot_1 = MLJ.transform(fit!(machine(Standardizer(), log.(select(train_df, ["Mid1", "Hexb", "Gm42418"]) .+ 1))))
pred_plot_1.labels = y
p = PlotlyJS.plot(pred_plot_1, x=:Mid1, y=:Hexb, z=:Gm42418, color=:labels, kind="scatter3d", mode="markers",  Layout(title="Mid1, Hexb, Gm42418 Log"))
#save as html file
open("./Plots/pred_plot_Mid1_Hexb_Gm42418_LOG.html", "w") do io PlotlyBase.to_html(io, p.plot) end