using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels, MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("./data_processing.jl")
include("./models.jl")
include("./data_analysis.jl")


#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

#Spliting, Normalizing data
normalised = true
y = coerce!(train_df, :labels => Multiclass).labels
if normalised
    x_train, x_test = clean(train_df, test_df)
    x_train, x_test = norm(x_train, x_test)
else
    x_train, x_test = clean(train_df, test_df)
end

x_train, x_test = clean(train_df, test_df)
x_train_preds, x_test_preds  = chose_predictors(x_train, x_test, 0.6)
x_train_norm, x_test_norm = norm(x_train_preds, x_test_preds)

m5 = machine(SVC(), x_train_norm[1:4000,:], y[1:4000]);
fit!(m5, verbosity = 0);
pred = predict(m5, x_train_norm[1:4000,:])
mean(pred.== y[1:4000])
pred = predict(m5, x_train_norm[4001:end,:])
mean(pred.== y[4001:end])




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
mach = machine(MultinomialClassifier(penalty = :none), x_train[1:4000,:], y[1:4000]) |> fit!
pred_train = predict_mode(mach, x_train[4001:end,:])
mean(pred_train.== y[4001:end])
"""

