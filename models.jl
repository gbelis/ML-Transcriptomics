using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics


function multinom_class(x_train, x_test, y)
    mach = machine(MultinomialClassifier(penalty = :none), x_train, y) |> fit!
    pred = predict_mode(mach, x_test)
    kaggle_submit(pred, "MultinomialClassifier_27_11")
    return
end

function multi_knn(x_train,y) #Trash
    model = MultitargetKNNClassifier(output_type = ColumnTable)
    self_tuning_model = TunedModel(model = model, resampling = CV(nfolds = 5), tuning = Grid(),
                        range = range(model, :K, values = 1:50), measure = MisclassificationRate)
    KNN_tuned = machine(self_tuning_model, x_train, y, scitype_check_level=0)
    fit!(KNN_tuned, verbosity = 0)
    
end