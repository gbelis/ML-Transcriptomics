using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


#Clean Data
y = coerce!(train_df, :labels => Multiclass).labels
x_train_clean, x_test_clean = clean(train_df, test_df)


x_train_preds, x_test_preds = chose_predictors(x_train_clean, x_test_clean, 0.25)
x_train, x_test = no_corr(x_train_preds, x_test_preds)
x_train_norm, x_test_norm = norm(x_train, x_test)








m5 = machine(SVC(), x_train_norm[1:4000,:], y[1:4000]);
fit!(m5, verbosity = 0);
pred = predict(m5, x_train_norm[1:4000,:])
mean(pred.== y[1:4000])
pred = predict(m5, x_train_norm[4001:end,:])
mean(pred.== y[4001:end])

m5 = machine(SVC(), x_train[1:4000,:], y[1:4000]);
fit!(m5, verbosity = 0);
pred = predict(m5, x_train[1:4000,:])
mean(pred.== y[1:4000])
pred = predict(m5, x_train[4001:end,:])
mean(pred.== y[4001:end])


Random.seed!(0)
svc = SVC()
mach = machine(TunedModel(model = svc,
                        resampling = CV(nfolds = 5),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 10),
                        range = [range(scv, :cost, lower = 100, upper = 1000, scale = :log10),
                                    range(svc, :gamma, lower = 1e-6, upper = 0.01, scale = :log10)]),
                x_train_norm, y) |> fit!

pred = predict(mach, x_test_norm)
kaggle_submit(pred, "SCV_Norm_3021preds_3_11_v2") 
rep = report(mach)
plot(mach)
pred = predict(mach, x_train_norm)
mean(pred.== y)
rep.best_model
"""
Random.seed!(0)
xgb = XGBoostClassifier()
mach = machine(TunedModel(model = xgb,
                        resampling = CV(nfolds = 4),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 4),
                        range = [range(xgb, :num_round, lower = 50, upper = 500),
                                    range(xgb, :max_depth, lower = 2, upper = 6)]),
                x_train, y) |> fit!
plot(mach)
report(mach)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "XGBoost_preds27_1_12_norm_v2")
"""

