using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using OpenML, MLJ,CSV, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, Plots, MLJFlux, Flux,DataFrames,  MLJMultivariateStatsInterface, Random, StatsPlots,MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=true, from_index=true)
train_df = nothing
test_df = nothing

pred_names = get_names_len(x_train, y, 6000)
x_train = select(x_train, pred_names)

x_test = select(x_test, names(x_train))
data= vcat(x_train,x_test)
data = MLJ.transform(fit!(machine(PCA(maxoutdim = 3000), data)), data)

Random.seed!(0)

xgb = XGBoostClassifier(max_depth=5, min_child_weight=4.33, subsample=0.75, colsample_bytree=0.75, gamma=0)
mach = machine(TunedModel(model = xgb,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 16),
                        range = [range(xgb, :eta, lower = 1e-2, upper = .2, scale = :log),
                                range(xgb, :num_round, lower = 50, upper = 500)]),
                                 data[1:5000,:], y)
fit!(mach)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "XGBClassifier_pca")
fitted_params(mach).best_model
plot(mach)

###################################################### Best Model

training_data, validation_data, test_data,validation_test = data_split(data[1:5000,:],y, 1:4000, 4001:5000, shuffle =true)
mach = machine(XGBoostClassifier(eta = 0.2, num_round=500, max_depth=5, min_child_weight=4.33, subsample=0.75, colsample_bytree=0.75, gamma=0), training_data, validation_data) |> fit!
m = mean(predict_mode(mach, test_data) .== validation_test)
