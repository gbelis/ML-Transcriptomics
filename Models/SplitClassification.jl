# This file is attempt to look at the classification as two different steps. The first predicting wether the cell is control (eGFP) or modified (CBP or KAT5).
# Then split KAT5 and CBP after. 

using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")
Random.seed!(0)


train_df = DataFrame(CSV.File("./data/train.csv.gz"))
test_df = DataFrame(CSV.File("./data/test.csv.gz"))


# Cleaning, Standardizing
y_1 = copy(train_df.labels)
y_1[y_1 .!= "eGFP"] .= "NOT"
y_1 = coerce!(DataFrame(Y = y_1), :Y => Multiclass).Y

y_1 = coerce!(train_df, :labels => Multiclass).labels


x_train, x_test = clean(train_df, test_df)
x_train, x_test = no_corr(x_train, x_test)
x_train_norm, x_test_norm = norm(x_train, x_test)

# Tuning a linear model for the first step
seed, goal, lower, upper = 0,5,1e-7,1e-3
Random.seed!(seed)
model = LogisticClassifier(penalty = :l1)
mach_lasso = machine(TunedModel(model = model,
                                resampling = CV(nfolds = 4),
                                tuning = Grid(goal = goal),
                                range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                measure = MisclassificationRate()),
                                x_train, y_1) |>fit!

pred = predict_mode(mach_lasso, x_train)
mean(pred .== y_1)
rep = report(mach_lasso)




y_2 = train_df.labels[train_df.labels .!= "eGFP"]
y_2 = coerce!(DataFrame(Y = y_2), :Y => Multiclass).Y



y_1[train_df.labels .!= "eGFP"] = y_2
y_1

train_df.labels