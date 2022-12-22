using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLCourse, Statistics, Distributions, OpenML, MLJMultivariateStatsInterface, MLJFlux, Flux, MLJLinearModels
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")
Random.seed!(0)

#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

# Cleaning, Standardizing, Chosing predictors
y = coerce!(train_df, :labels => Multiclass).labels
x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
x_train_norm, x_test_norm = norm(x_train, x_test)


names_sel = get_names_len(x_train_norm, y, 6000)
x_train_sel = select(x_train_norm, names_sel)
x_test_sel = select(x_test_norm, names_sel)



data= vcat(x_train_sel,x_test_sel)
mach_pca = fit!(machine(PCA(), data))
vars = report(mach_pca).principalvars ./ report(mach_pca).tvar
p2 = plot(cumsum(vars), label = nothing, xlabel = "component", ylabel = "cumulative prop. of variance explained")

data_tranformed = MLJ.transform(fit!(machine(PCA(maxoutdim = 3000), data)), data)

x_train_NN = data_tranformed[1:5000,:]
x_test_NN = data_tranformed[5001:end,:]


#models
builder = MLJFlux.Short(n_hidden = 50).dropout

model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100), batch_size = 256)
tunedNN = machine(TunedModel(model = model,
								tuning = Grid(goal = 9),
						        resampling = CV(nfolds = 4),
	                            range = [range(model,
									:(builder.dropout), lower = .5, upper = .8, scale = :linear),
								   range(model, :epochs, lower = 50, upper = 200, scale = :linear)],
	                          measure = MisclassificationRate()), x_train_NN, y) |>fit!

plot(tunedNN)
rep = report(tunedNN)
rep.best_model

pred = predict_mode(tunedNN, x_test_NN)
kaggle_submit(pred, "NNC_1x100_ep100_drop06_bs256_15_12.csv")



# ==================================================================
# 500 preds

names_sel = get_names_len(x_train_norm, y, 500)
x_train_sel = select(x_train_norm, names_sel)
x_test_sel = select(x_test_norm, names_sel)
names_sel

model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100), batch_size = 256)
tunedNN = machine(TunedModel(model = model,
						        resampling = CV(nfolds = 5),
								tuning = Grid(goal = 36),
	                            range = [range(model, :(builder.dropout), lower = .0, upper = .9, scale = :linear),
								        range(model, :epochs, lower = 50, upper = 250, scale = :linear)],
								   measure = MisclassificationRate()), x_train_sel, y) |>fit!

rep = report(tunedNN)
rep.best_model
rep.history
plot(tunedNN)
mean(predict_mode(tunedNN, x_train_sel) .== y)

# ==================================================================
# Supervised UMAP
train_umap_df = DataFrame(CSV.File("./data/train_umap.csv"))[:,2:end]
test_umap_df = DataFrame(CSV.File("./data/test_umap.csv"))[:,2:end]
train_umap_df

model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100), batch_size = 256)
tunedNN = machine(TunedModel(model = model,
						        resampling = CV(nfolds = 5),
								tuning = Grid(goal = 10),
	                            range = range(model, :(builder.dropout), lower = .0, upper = .9, scale = :linear),
								   measure = MisclassificationRate()), x_train_sel, y) |>fit!
rep = report(tunedNN)
rep.best_model
rep.history
plot(tunedNN)
mean(predict_mode(tunedNN, x_train_sel) .== y)

pred = predict_mode(tunedNN, x_test_sel)
kaggle_submit(pred, "NNShort_500preds_100h_170ep_256bs_09do")



dummy_model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100, dropout = 0), batch_size = 256, epochs = 200)
dummy_mach = machine(dummy_model, x_train_sel, y) |>fit!

MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))).dropout



model = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))))
tunedNN = TunedModel(model = model,
                                goal = 9,
						        resampling = CV(nfolds = 5),
	                            range = [range(model,
									:(neural_network_classifier.builder.dropout),
									values = [0., .1, .2, ]),
								   range(model,
									 :(neural_network_classifier.epochs),
									 values = [500, 1000, 2000])],
	                          measure = MisclassificationRate(), x_train_sel, y) |>fit!

pred = predict_mode(tunedNN, x_test_sel)
kaggle_submit(pred, "NNC_1x100_eptuned.csv")