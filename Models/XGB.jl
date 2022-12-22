using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using OpenML, MLJ,CSV, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, Plots, MLJFlux, Flux,DataFrames,  MLJMultivariateStatsInterface, Random, StatsPlots,MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")

"""
XGBoost Classification using cross-validation. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data

Hyperparameters tuned : num_round [50,500], max_depth [2,10], eta [1e-2,0.2], min_child_weight [1,6], subsample [0.6,0.9], colsample_bytree [0.6,0.9], gamma [0,0.4]
"""

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], 55)
indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],55)
indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],55)
indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
x_train = select(x_train, indexes)
x_test = select(x_test, indexes)

#standardize data
x_train, x_test = norm(x_train, x_test)
train_df = nothing
test_df = nothing

##feature selection
pred_names = get_names_len(x_train, y, 6000)
x_train = select(x_train, pred_names)
x_test = select(x_test, names(x_train))

#pca
data= vcat(x_train,x_test)
data = MLJ.transform(fit!(machine(PCA(maxoutdim = 3000), data)), data)

###################################################### Tuning

Random.seed!(0) #fix seed

xgb = XGBoostClassifier(max_depth=5, min_child_weight=4.33, subsample=0.75, colsample_bytree=0.75, gamma=0) #previously tuned
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

##################################################### Best Model

training_data, validation_data, test_data,validation_test = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(XGBoostClassifier(eta = 0.47, num_round=500, max_depth=7, min_child_weight=4.33, subsample=0.75, colsample_bytree=0.75, gamma=0), training_data, validation_data) |> fit!
m = mean(predict_mode(mach, test_data) .== validation_test)

mach = machine(XGBoostClassifier(eta = 0.47, num_round=500, max_depth=7, min_child_weight=4.33, subsample=0.75, colsample_bytree=0.75, gamma=0), x_train, y) |> fit!
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "XGB_call_rates")