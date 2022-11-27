using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics


function multinom_class(x_train, x_test, y)
    mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
    pred = predict_mode(mach, x_test)
    kaggle_submit(pred, "MultinomialClassifier_27_11")
    return
end

