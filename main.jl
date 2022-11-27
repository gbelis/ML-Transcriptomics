using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels
include("./utility.jl")
include("./models.jl")

train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


x_train = clean_data(select(train_df, Not(:labels)))
x_test = clean_data(select(test_df, names(x_train)))
x_train = select(train_df, names(x_test))

y = coerce!(train_df, :labels => Multiclass).labels

"""
mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
pred_train = predict_mode(mach, x_train)
mean(pred_train  .== y)
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "MultinomialClassifier_27_11_v2")
"""


"""
model = MultitargetKNNClassifier(output_type = ColumnTable)
self_tuning_model = TunedModel(model = model, resampling = CV(nfolds = 5), tuning = Grid(),
                    range = range(model, :K, values = 1:50), measure = MisclassificationRate)

KNN_tuned = machine(self_tuning_model, x_train, y, scitype_check_level=0)
fit!(KNN_tuned, verbosity = 0)

"""