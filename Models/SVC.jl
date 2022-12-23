using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLCourse, Statistics, Distributions,OpenML, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("../data_processing.jl")
include("../data_analysis.jl")
include("../models.jl")

#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


#Clean Data
y = coerce!(train_df, :labels => Multiclass).labels
x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
x_train_norm, x_test_norm = norm(x_train, x_test)

x_train_sel = correlation_labels(x_train_norm, 1122)
n_1200 = 1122
n_5000 = 2902
n_6000 = 3529

x_test_sel = select(x_test, names(x_train_sel))


Random.seed!(0)
svc = SVC(gamma = 2e-4)
mach = machine(TunedModel(model = svc,
                        resampling = CV(nfolds = 3),
                        measure = MisclassificationRate(),
                        tuning = Grid(goal = 9),
                        range = range(svc, :cost, lower = 100, upper = 1e4, scale = :log10)),
                x_train_sel, y) |> fit!

pred = predict(mach, x_test_norm)
kaggle_submit(pred, "SCV_Norm_3021preds_3_11_v2") 
rep = report(mach)
plot(mach)
pred = predict(mach, x_train_sel)
mean(pred .== y)
rep.best_model