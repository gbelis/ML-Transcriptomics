# Best result using a linear method, a Lasso Classifier

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("../data_processing.jl")
include("../models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

#Previously tuned to find an interval of the best lambda
seed, goal, lower, upper = 0,9, 6e-5, 1e-4

Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)
mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper),
                                measure = MisclassificationRate()),
                                x_train, y) |>fit!
fitted_params(mach_lasso).best_model
report(mach_lasso)
pred = predict_mode(mach_lasso, x_test)
kaggle_submit(pred, "LassoClassifier_12")