using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML
include("./utility.jl")
include("./models.jl")

train_df = DataFrame(CSV.File("train.csv.gz"))

train_data, train_prediction = separate_data_prediction(train_df)
train_df_clean = clean_data(train_data)
coerce!(train_df_clean, :labels => Multiclass)

#mach = machine(MultinomialClassifier(penalty = :none), select(df_no_zero, not(:)), df_no_zero.labels[1:100]) |> fit!