using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


#Clean Data tu peux faire ce que tu veux mais t'as tord mes fonctions sont mieux
y = coerce!(train_df, :labels => Multiclass).labels
x_train_clean, x_test_clean = clean(train_df, test_df)


# Cross Validation
results = DataFrame(t_cutoff = 0., accuracy = 0.)   # La premiere ligne est nulle mais jai pas trouve comment faire autrement
X = x_train # tout le training data

goal, n_folds, lower, upper = 5, 1, 0, 4  #ca tu comprends
for i in range(lower, upper, goal)
    m = 0.0
    for j in range(1, n_folds, n_folds)
        println("completed $(i*j) trainings out of $(goal*n_folds), $(i*j/goal/n_folds)%")
        train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
        
        mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y)
        fit(verbosity = 0)!

        m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
    end
    push!(results, [i, m/n_folds])
end

range(1, 5, 5)