using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("./utility.jl")
include("./models.jl")

train_df = DataFrame(CSV.File("./data/test.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))

x = clean_data(vcat(select(train_df, Not(:labels)), test_df))
y = coerce!(train_df, :labels => Multiclass).labels
x_train = x[1:5000,:]
x_test = x[5001:end,:]

mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
pred_train = predict_mode(mach, x_train)
mean(pred_train  .== y)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "MultinomialClassifier_27_11")