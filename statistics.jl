using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML, NearestNeighborModels,
MLJXGBoostInterface, MLJDecisionTreeInterface, MLJMultivariateStatsInterface, MLJLIBSVMInterface
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")


#Importing Data
train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


y = coerce!(train_df, :labels => Multiclass).labels
x_train = remove_constant_predictors(select(train_df, Not(:labels)))
x_train = remove_prop_predictors(x_train)

#Fitting MultinomialClassifier to get 
mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
report(mach)
params = fitted_params(mach)
params



levels(params.classes)[1]
titles = DataFrame(titles = levels(params.classes))
coefs = DataFrame(params.coefs)
intercept = DataFrame(intercept = params.intercept)
beta_df = hcat(titles, coefs)
beta_df[1,:]



# CBP
beta_df
CBP_df = DataFrame(beta_df[1, :])
CBP_df = permutedims(CBP_df, 1)
CBP_df.stds = std.(eachcol(x_train))
CBP_df.t_value = CBP_df[:,2] ./ CBP_df.stds
CBP_df.abs_t = abs.(CBP_df.t_value)
CBP_df

CBP_df[:,2]

select(CBP_df, Not([:titles, :stds])) ./ CBP_df.stds

select(CBP_df, Not([:titles, :stds]))

# KAT5
KAT5_df = DataFrame(beta_df[2, :])
KAT5_df = permutedims(KAT5_df, 1)
KAT5_df.stds = std.(eachcol(x_train))
KAT5_df.t_value = KAT5_df.KAT5 ./ KAT5_df.stds
KAT5_df.abs_t = abs.(KAT5_df.t_value)
KAT5_df

# eGFP
eGFP_df = DataFrame(beta_df[3, :])
eGFP_df = permutedims(eGFP_df, 1)
eGFP_df.stds = std.(eachcol(x_train))
eGFP_df.t_value = eGFP_df.eGFP ./ eGFP_df.stds
eGFP_df.abs_t = abs.(eGFP_df.t_value)
eGFP_df

#t_vals
t_vals = DataFrame(genes = names(beta_df[:,2:end]), CBP = CBP_df.abs_t, KAT5 = KAT5_df.abs_t, eGFP = eGFP_df.abs_t)
t_vals = permutedims(t_vals, 1)
t_vals


maxs = DataFrame(genes = names(t_vals[:,2:end]) ,maxs = maximum.(eachcol(t_vals[:,2:end])))
chosen = maxs[maxs.maxs .> 1, :]
chosen_names = names(permutedims(chosen, 1)[:,2:end])

#Cross validation to chose t-value cutoff
info = DataFrame(t_cutoff = 0., accuracy = 0.)

y = coerce!(train_df, :labels => Multiclass).labels
goal, n_folds, lower, upper = 50, 5, 0, 4

for i in range(lower, upper, goal)
    chosen_names = names(permutedims(maxs[maxs.maxs .> i, :], 1)[:,2:end])
    X_chosen = select(train_df, chosen_names)
    m = 0.0
    for j in range(0,0,n_folds)
        train_x_chosen, train_y_chosen, test_x_chosen, test_y_chosen = data_split(X_chosen,y, 1:4000, 4001:5000)
        mach_chosen = machine(MultinomialClassifier(penalty = :none), train_x_chosen, train_y_chosen) |> fit!
        m += mean(predict_mode(mach_chosen, test_x_chosen) .== test_y_chosen)
    end
    push!(info, [i, m/n_folds])
end


info[14,2]
CSV.write("./data/info_on_regression_tuning.csv", info)

scatter(info.t_cutoff, info.accuracy, label = "data points")


X = remove_constant_predictors(select(train_df, Not(:labels)))
y = coerce!(train_df, :labels => Multiclass).labels
train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y) |> fit!
mean(predict_mode(mach, test_x) .== test_y)

Random.seed!(0)
X_chosen = select(train_df, chosen_names)
y = coerce!(train_df, :labels => Multiclass).labels
train_x_chosen, train_y_chosen, test_x_chosen, test_y_chosen = data_split(X_chosen,y, 1:4000, 4001:5000)
mach_chosen = machine(MultinomialClassifier(penalty = :none), train_x_chosen, train_y_chosen) |> fit!
mean(predict_mode(mach_chosen, test_x_chosen) .== test_y_chosen)


Random.seed!(0)
X_train_sub = select(train_df, chosen_names)
X_test_sub = select(test_df, chosen_names)
y = coerce!(train_df, :labels => Multiclass).labels
mach_sub = machine(MultinomialClassifier(penalty = :none), X_train_sub, y) |> fit!
pred = predict_mode(mach_sub, X_test_sub)
kaggle_submit(pred, "Multinomial_tval1_8493preds_7_12")  





# population size = 5000
# signification level alpha = 0.05 
t_0_05 = 1.960439
t_0_01 = 2.576813



#try not to overfit


function coef_info(beta_df, x_train)
    df = permutedims(beta_df, 1)
    df.stds = std.(eachcol(x_train))
    df.t_value = df[:,2] ./ df.stds
    df.abs_t = abs.(df.t_value)
    return df
end

function get_names(X, y)
    mach = machine(MultinomialClassifier(penalty = :none), X, y) |> fit!
    params = fitted_params(mach)
    df = hcat(DataFrame(titles = levels(params.classes)), DataFrame(params.coefs))
    info = DataFrame(genes = names(beta_df[:,2:end]))
    for i in range(1,3,3)
        info.levels(params.classes)[i] = coef_info(DataFrame(beta_df[i, :]), X)
    end
    
    return
end


X = x_train

for i in range(lower, upper, goal)
    m = 0.0
    for j in range(0,0,n_folds)
        train_x, train_y, test_x, test_y = data_split(X,y, 1:4000, 4001:5000)
        pred_names = get_names(train_x, train_y)
        train_x = select(train_x, pred_names)
        mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y) |> fit!
        m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
    end
    push!(info, [i, m/n_folds])
end