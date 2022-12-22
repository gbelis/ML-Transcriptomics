# This file contains code for all figures 

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
import Pkg; Pkg.add("PlotlyJS")
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface, MLJFlux, Flux, UMAP, PlotlyJS
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")


#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

y = coerce!(train_df, :labels => Multiclass).labels
x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
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



x_train_3 = correlation_labels(x_train_norm, 3)
x_train_3.y = y
x_train_3
PlotlyJS.plot(x_train_3, x=:Hexb, y=:Gm26917, z=:Lrp1, color=:y, kind="scatter3d", mode="markers")



x_train_3_log_norm = correlation_labels(x_train_log, 3)
x_train_3_log_norm.y = y
x_train_3_log_norm
PlotlyJS.plot(x_train_3_log_norm, x=:Gm42418, y=:Gm26917, z=:Hexb, color=:y, kind="scatter3d", mode="markers")


x_train_3_log = correlation_labels(log2.(x_train.+1), 3)
x_train_3_log.y = y
x_train_3_log
PlotlyJS.plot(x_train_3_log, x=:Gm42418, y=:Gm26917, z=:Hexb, color=:y, kind="scatter3d", mode="markers")