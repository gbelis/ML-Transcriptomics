using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface, MLJFlux, Flux
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")


#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

y = coerce!(train_df, :labels => Multiclass).labels
x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
chosen_names = get_names(x_train, y, 2)
x_train, x_test = select(x_train, chosen_names), select(x_test, chosen_names)
x_train, x_test = norm(x_train, x_test)


results = DataFrame(t_cutoff = 0., call_rates = 0., accuracy = 0.)   # La premiere ligne est nulle mais jai pas trouve comment faire autrement
X = x_train # tout le training data


goal, n_folds = 5, 1
# T value selection
lower_t, upper_t = 0, 4  
# Call rates selection
lower_r, upper_r = 0, 100
for r in range(lower, upper, goal)
    for t in range()
        m = 0.0
        for j in range(1, n_folds, n_folds)
            println("completed $(i*j) trainings out of $(goal*n_folds), $(i*j/goal/n_folds) %")
            train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
            
            mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y)
            fit(verbosity = 0)!

            m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
        end
        push!(results, [i, m/n_folds])
    end
end




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


                                    tunedNN = TunedModel(model = model,
                                goal = 10,
						        resampling = CV(nfolds = 5),
	                            range = range(model, :epochs,  lower = 10, upper = 1000, scale = log10),
	                          measure = MisclassificationRate(), x_train, y) |>fit!

pred = predict_mode(tunedNN, x_test)
kaggle_submit(pred, "NNC_2x100_bs128_eptuned.csv")





mach = machine(NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                    Dense(100, n_out))),
                                    batch_size = 128,
                                    epochs = 100),
                                    x_train,
                                    y)
fit!(mach)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "NNC_2x100_bs128_ep100.csv")
