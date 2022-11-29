using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics


function multinom_class(x_train, x_test, y, pena)
    mach = machine(MultinomialClassifier(penalty = pena), x_train, y) |> fit!
    pred = predict_mode(mach, x_test)
    kaggle_submit(pred, "MultinomialClassifier_$(pena)_29_11")
end


function lasso_classifier(x_train, x_test, y, seed, goal, lower, upper)
    Random.seed!(seed)
    model = MultinomialClassifier(penalty = :l1)
    mach_lasso = machine(TunedModel(model = model,
                                    resampling = CV(nfolds = 5),
                                    tuning = Grid(goal = goal),
                                    range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                    measure = MisclassificationRate()),
                                    x_train, y) |>fit!
    pred = predict_mode(mach_lasso, x_test)
    kaggle_submit(pred, "LassoClassifier_seed$(seed)_29_11")  
    return mach_lasso
end

function ridge_classifier(x_train, x_test, y, seed, goal, lower, upper)
    Random.seed!(seed)
    model = MultinomialClassifier(penalty = :l2)
    tuned_model_ridge = TunedModel(model = model,
                                    resampling = CV(nfolds = 5),
                                    tuning = Grid(goal = goal),
                                    range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                    measure = MisclassificationRate())
    mach_ridge = machine(tuned_model_ridge, x_train, y) |>fit!
    pred = predict_mode(mach_ridge, x_test)
    kaggle_submit(pred, "RidgeClassifier_28_11")  
    return mach_ridge
end
   
function XGboost_class(x_train, x_test, y)

    pipe = @pipeline(Standardizer, OneHotEncoder)#, cfr())
    pipe.xg_boost_classifier.num_round = 200
    pipe.xg_boost_classifier.subsample = 0.8
    pipe.xg_boost_classifier.colsample_bytree = 0.8
    mach = machine(TunedModel(model = pipe,
                            resampling = CV(nfolds = 4),
		                    measure = MisclassificationRate(),
                            tuning = Grid(goal = 25),
                            range = [range(pipe, :eta, lower = 1e-2, upper = .1, scale = :log),
                                     range(pipe, :num_round, lower = 50, upper = 500),
                                     range(pipe, :max_depth, lower = 2, upper = 6)]),
                  x_train, y)
	fit!(mach, verbosity = 0)
   
    kaggle_submit(pred, "MultinomialClassifier_$(pena)_27_11")
end


    #r = [
    #    range(pipe, :(xg_boost_classifier.eta), lower=0.001, upper=0.03, scale=:log),
    #    range(pipe, :(xg_boost_classifier.max_depth), lower=2, upper=10)
    #    ]
    
    #pipe.xg_boost_classifier.num_round = 200
    #pipe.xg_boost_classifier.subsample = 0.8
    #pipe.xg_boost_classifier.colsample_bytree = 0.8
    #self_tuning_xgb = TunedModel(model=pipe,
    #                              resampling=CV(nfolds=3),
    #                              tuning=RandomSearch(),
    #                              range=r,
    #                              measure=[LogLoss()],n=2000);
    #m = machine(self_tuning_xgb, X, y)
    #fit!(m, rows=train)
    #fitted_params(m).best_model

function multi_knn(x_train,y) #Trash
    model = MultitargetKNNClassifier(output_type = ColumnTable)
    self_tuning_model = TunedModel(model = model, resampling = CV(nfolds = 5), tuning = Grid(),
                        range = range(model, :K, values = 1:50), measure = MisclassificationRate)
    KNN_tuned = machine(self_tuning_model, x_train, y, scitype_check_level=0)
    fit!(KNN_tuned, verbosity = 0)
    
end

