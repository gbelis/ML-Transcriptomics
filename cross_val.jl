# This file contains a TEMPLATE for cross validation. This allows us to combine feature selection tuning with model specific hyperparameter tuning.

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


#Cleaning Data
y = coerce!(train_df, :labels => Multiclass).labels
x_train_clean, x_test_clean = clean(train_df, test_df)


# Cross Validation
results = DataFrame(t_cutoff = 0., accuracy = 0.)   # La premiere ligne est nulle mais jai pas trouve comment faire autrement
X = x_train # tout le training data


# ==============  DOES NOT RUN ON ITS OWN =======================
# ===================  MUST ADD MODEL ===========================
# One Hyperparameter
goal, n_folds, lower, upper = 5, 1, 0, 4 
for i in range(lower, upper, goal)
    m = 0.0
    for j in range(1, n_folds, n_folds)
        train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
        
        #model = ...
        mach = machine(model, train_x, train_y)
        fit(verbosity = 0)!

        m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
    end
    push!(results, [i, m/n_folds])
end


# Two Hyperparameters

goal, n_folds = 5, 1
# T value selection
lower_1, upper_1 = 0, 4  
# Call rates selection
lower_2, upper_2 = 0, 100
for t in range(lower_1, upper_1, goal)
    for r in range(lower_2, upper_2, goal)
        for t in range()
            m = 0.0
            for j in range(1, n_folds, n_folds)
                train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
                
                # mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y)
                fit(verbosity = 0)!

                m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
            end
            push!(results, [i, m/n_folds])
        end
    end
end
