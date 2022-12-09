using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, Plots, MLJFlux, Flux,DataFrames
include("../data_processing.jl")
include("../data_analysis.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, normalised=true, from_index=true)



machl = lasso_classifier(x_train, x_test, y, seed=0, goal=10, lower=1e-5, upper=1e-4)

pred = predict_mode(machl, data8[5001:8093,:])
best = DataFrame(params = fitted_params(machl).best_model)
best = DataFrame(params = fitted_params(machl).best_model)
CSV.write("./data/bestmodel.csv",best)
plot(machl)


Random.seed!(0)
# function fit_and_evaluate2(training_data, validation_data, test_data,validation_test)
#     xgb = XGBoostClassifier()
#     mach = machine(TunedModel(model = xgb,
#                             resampling = CV(nfolds = 4),
#                             measure = rmse,
#                             tuning = Grid(goal = 25),
#                             range = [range(xgb, :eta,
#                                            lower = 1e-2, upper = .1, scale = :log),
#                                      range(xgb, :num_round, lower = 50, upper = 500),
#                                      range(xgb, :max_depth, lower = 2, upper = 6)]),
#                                      training_data, validation_data)
#     mach = machine(XGBoostClassifier(), training_data, validation_data)
#     fit!(mach, verbosity = 0)
#     return mach
# end

# t,tv,te,tev = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
# mach3=fit_and_evaluate2(t,tv,te,tev)
# DataFrame(trainin_error = mean(predict_mode(mach3, t) .!= tv), test_error = mean(predict_mode(mach3, te) .!= tev))

#data = load_data("./data/CSV_8000.csv.gz")

xgb = XGBoostClassifier()
mach = machine(TunedModel(model = xgb,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 10),
                        range = range(xgb, :eta, lower = 1e-2, upper = .2, scale = :log)),
                                 x_train, y)
fit!(mach)
pred = predict_mode(mach, x_test)
printl(pred)
kaggle_submit(pred, "XGB_pca8000")
fitted_params(mach).best_model
plot(mach)

CSV.write("./data/bestmodel.csv",DataFrame(params = fitted_params(mach).best_model))