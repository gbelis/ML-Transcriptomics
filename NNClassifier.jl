using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLCourse, Statistics, Distributions, OpenML, MLJMultivariateStatsInterface, MLJFlux, Flux
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