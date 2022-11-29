using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels
include("./utility.jl")
include("./models.jl")

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


"""
mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
pred_train = predict_mode(mach, x_train)
mean(pred_train  .== y)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "MultinomialClassifier_29_11_norm_v2")
"""

"""
x_train = clean_data(select(train_df, Not(:labels)))
x_test = clean_data(select(test_df, names(x_train)))
x_train = select(train_df, names(x_test))

"""