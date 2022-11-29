using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels
include("./data_processing.jl")
include("./models.jl")
include("./data_analysis.jl")


train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

x_train,x_test,y = clean_data(train_df, test_df)

#mach1 = lasso_classifier(x_train[1:100,1:100], x_test[1:100,1:100], y[1:100], 0, 15, 1e-5, 5e-4)

mach1= ridge_classifier(x_train[1:100,1:100], x_test[1:100,1:100], y[1:100], 0, 10, 1e-6, 1e+2)
plot(mach1)
confusion_matrix(predict_mode(mach1),y[1:100] )

