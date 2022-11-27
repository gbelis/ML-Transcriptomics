using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("./utility.jl")
include("./models.jl")



train_df = DataFrame(CSV.File("/Users/guillaumebelissent/Docs/EPFL/BA5/ML/Project/train.csv.gz"))
test_df = DataFrame(CSV.File("/Users/guillaumebelissent/Docs/EPFL/BA5/ML/Project/test.csv.gz"))

x = clean_data(vcat(select(train_df, Not(:labels)), test_df))
y = coerce!(train_df, :labels => Multiclass).labels
x_train = x[1:5000,:]
x_pred = x[5001:end,:]

mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
pred_train = predict_mode(mach, x_train)
mean(pred_train  .== y)

pred = predict_mode(mach, x_pred)