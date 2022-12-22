using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
Pkg.add("BSON")
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface, MLJFlux, Flux, BSON
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



mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))
mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))
mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))

data = DataFrame(genes = names(x_train), diff = abs.(mean_CBP-mean_KAT5))
sort!(data, :diff, rev = true)
data
sel_names = data.genes[1:1000]

PlotlyJS.plot(train_df, x=:Mid1, y=:Polr1b, z=:Lrp1, color=:labels, kind="scatter3d", mode="markers")


train_not_eGFP = train_df[y .!= "eGFP",:]
select!(train_not_eGFP, sel_names)

X, Y, X_test, Y_test = data_split(train_not_eGFP,y[y .!= "eGFP"], 1:2000, 2001:3408)
mach = machine(LogisticClassifier(penalty = :none),X, Y) |> fit!
pred = predict_mode(mach, X_test)
mean(pred .== Y_test)

# =======================================================================================================================================

#models
mach = machine(NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                    Dense(100, 100, relu), Dense(100, n_out))),
                                    batch_size = 128,
                                    epochs = 100),
                                    x_train, y)

fit!(mach)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "NNC_3x100_bs128_ep100")

mean(predict_mode(mach, x_train) .== y)
# =======================================================================================================================================




model = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))))
tunedNN = TunedModel(model = model,
                                goal = 10,
						        resampling = CV(nfolds = 5),
	                            range = [range(model,
									:(neural_network_regressor.builder.dropout),
									values = [0., .1, .2, ]),
								   range(model,
									 :(neural_network_regressor.epochs),
									 values = [500, 1000, 2000])],
	                          measure = MisclassificationRate(), x_train, y) |>fit!

pred = predict_mode(tunedNN, x_test)
kaggle_submit(pred, "NNC_2x100_bs128_eptuned.csv")


# =======================================================================================================================================


mach = machine(NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                    Dense(100, n_out))),
                                    batch_size = 128,
                                    epochs = 100),
                                    x_train,
                                    y)
fit!(mach)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "NNC_2x100_bs128_ep100.csv")
