# This is the file used for the tuning of neural networks.

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLCourse, Statistics, Distributions, OpenML, MLJMultivariateStatsInterface, MLJFlux, Flux, MLJLinearModels
include("../data_processing.jl")
include("../models.jl")
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
# Supervised UMAP --- Massively overfit the data
train_umap_df = DataFrame(CSV.File("./data/train_umap.csv"))[:,2:end]
test_umap_df = DataFrame(CSV.File("./data/test_umap.csv"))[:,2:end]
train_umap_df

model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100), batch_size = 256)
tunedNN = machine(TunedModel(model = model,
						        resampling = CV(nfolds = 5),
								tuning = Grid(goal = 10),
	                            range = range(model, :(builder.dropout), lower = .0, upper = .9, scale = :linear),
								   measure = MisclassificationRate()), train_umap_df, y) |>fit!
rep = report(tunedNN)
rep.best_model
rep.history
plot(tunedNN)
mean(predict_mode(tunedNN, train_umap_df) .== y)

pred = predict_mode(tunedNN, test_umap_df)
kaggle_submit(pred, "UMAPSup100_NN_ep10_do03_bs256")



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



# ======== USING DIY TUNING ==============



results = DataFrame([[],[]], ["dropout", "accuracy"])


X = x_train_norm # tout le training data

goal, n_folds, lower, upper = 10, 4, 0, 0.9  #ca tu comprends
iter = 0
for i in range(lower, upper, goal)
    m = 0.0
    for j in range(1, n_folds, n_folds)
        println("completed $(iter += 1) trainings out of $(goal*n_folds), $(iter/goal/n_folds*100)%")

		#Splitting Data
        train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)

		# Getting predictors
		names_sel = get_names_len(train_x, train_y, 6000)
		x_train_sel = select(train_x, names_sel)
		x_test_sel = select(test_x, names_sel)

		mach_pca = fit!(machine(PCA(maxoutdim = 3000), x_train_sel))
		x_train = MLJ.transform(mach_pca,x_train_sel)
		x_test = MLJ.transform(mach_pca,x_test_sel)


		# Creating and Fitting machine
        model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 500, dropout = i), batch_size = 256, epochs = 50)
        mach = machine(model, x_train, train_y)
        fit!(mach, verbosity = 1)

		# Evaluating 
        m += mean(predict_mode(mach, x_test) .== test_y)
    end
    push!(results, [i, m/n_folds])
end


scatter(results.dropout, results.accuracy)