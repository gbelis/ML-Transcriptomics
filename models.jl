using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse, Statistics

function fit_and_evaluate(training_data, validation_data, test_data,validation_test)
    """
        fit the data with a multinomailClassifier and evaluate it.

    Arguments:
        training_data {DataFrame} -- training data
        valididation_data {DataFrame} -- training labels
        test_data {DataFrame} -- test data
        validation_test {DataFrame} -- test label

    Returns:
        mach {machine} -- trained machine
        error {DataFrame} -- dataframe of training and test error

    """
    mach = machine(MultinomialClassifier(penalty = :l2, lambda = 1e-4), training_data, validation_data) |> fit!
    return mach, DataFrame(training_error = sum(MLJ.predict(mach).== validation_data),
    test_error = sum(MLJ.predict(mach, test_data).== validation_test), errorate = mean(predict_mode(mach, test_data) .!= validation_test))
end

function data_split(data,y, idx_train, idx_test; shuffle =true)
    """
        Split data between a train and test set

    Arguments:
        data {DataFrame} -- all the data to split
        y {DataFrame} -- labels of the data
        idx_train {UnitRange{Int64}} -- indexes of train data
        idx_test {UnitRange{Int64}} -- indexes of test data
        shuffle {boolean} -- if true shuffle the data

    Returns:
        train {DataFrame} -- training data
        train_valid {DataFrame} -- training labels
        test {DataFrame} -- test data
        test_valid {DataFrame} -- test label

    """
    if shuffle
        idxs = randperm(size(data, 1))
    else
        idxs= 1:size(data, 1)
    end
    return (train = data[idxs[idx_train], :],
    train_valid = y[idxs[idx_train], 1],
    test = data[idxs[idx_test], :],
    test_valid = y[idxs[idx_test], 1])
    end

function multinom_class(x_train, x_test, y; pena, lambda, title)
    """
        Multinomial Classification. Save the prediction in a csv file.

    Arguments:
        x_train {DataFrame} -- train set (without labels)
        x_test {DataFrame} -- test set to predict
        y {DataFrame} -- labels of the training data
        pena {} -- penalty

    Returns:
        mach{} -- machine

    """
    #mach = machine(MultinomialClassifier(penalty = pena, lambda = lambda), x_train, y) |> fit!
    mach = machine(LogisticClassifier(penalty = pena, lambda = lambda), x_train, y) |> fit!
    pred = predict_mode(mach, x_tests)
    kaggle_submit(pred, "MultinomialClassifier_pca2_$(pena)_$(title)_30_11")
    return mach
end


function lasso_classifier(x_train, x_test, y;seed=0, goal, lower, upper)
    """
        Lasso Classification using cross-validation. Save the prediction in a csv file.

    Arguments:
        x_train {DataFrame} -- train set (without labels)
        x_test {DataFrame} -- test set to predict
        y {DataFrame} -- labels of the training data
        seed {int} -- value of the seed to fix
        goal {int} -- number of different lambda to try
        lower {float} -- value of the smallest lambda to try
        upper {float} -- value of the biggest lambda to try

    Returns:
        mach{} -- machine

    """
    Random.seed!(seed)
    model = MultinomialClassifier(penalty = :l1)
    mach_lasso = machine(TunedModel(model = model,
                                    resampling = CV(nfolds = 5),
                                    tuning = Grid(goal = goal),
                                    range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                    measure = MisclassificationRate()),
                                    x_train, y) |>fit!
    pred = predict_mode(mach_lasso, x_test)
    kaggle_submit(pred, "LassoClassifier_30_11")  
    return mach_lasso
end

function ridge_classifier(x_train, x_test, y; seed=0, goal, lower, upper)
    """
        Ridge Classification using cross-validation. Save the prediction in a csv file.

    Arguments:
        x_train {DataFrame} -- train set (without labels)
        x_test {DataFrame} -- test set to predict
        y {DataFrame} -- labels of the training data
        seed {int} -- value of the seed to fix
        goal {int} -- number of different lambda to try
        lower {float} -- value of the smallest lambda to try
        upper {float} -- value of the biggest lambda to try

    Returns:
        mach{} -- machine

    """
    Random.seed!(seed)
    model = MultinomialClassifier(penalty = :l2)
    tuned_model_ridge = TunedModel(model = model,
                                    resampling = CV(nfolds = 5),
                                    tuning = Grid(goal = goal),
                                    range = range(model, :lambda, lower = lower, upper = upper, scale = :log10),
                                    measure = MisclassificationRate())
    mach_ridge = machine(tuned_model_ridge, x_train, y) |>fit!
    pred = predict_mode(mach_ridge, x_test)
    kaggle_submit(pred, "RidgeClassifier_30_11")  
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