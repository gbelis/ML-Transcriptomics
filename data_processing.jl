function load_data(path)
    return DataFrame(CSV.File(path))
end

function remove_constant_predictors(df)
    df_no_const = df[:,  std.(eachcol(df)) .!= 0]
    return df_no_const
end

function remove_prop_predictors(df)
end

function clean_data(train_df, test_df)
    x_train = remove_constant_predictors(select(train_df, Not(:labels)))
    x_test = remove_constant_predictors(select(test_df, names(x_train)))
    x_train = select(train_df, names(x_test))
    y = coerce!(train_df, :labels => Multiclass).labels
    return x_train,x_test,y
end

function standardize(df)
    mach = fit!(machine(Standardizer(), df));
    data = MLJ.transform(mach, df)
    return data
end

function kaggle_submit(df_prediction, title)
    prediction_kaggle = DataFrame(id = collect(1:length(df_prediction)))
    prediction_kaggle[!,:prediction] = df_prediction
    CSV.write("./Submission/$(title).csv", prediction_kaggle)
end

function clean(train_df, test_df)
    x_train = clear_const(select(train_df, Not(:labels)))
    x_test = clear_const(select(test_df, names(x_train)))
    x_train = select(train_df, names(x_test))
    return x_train, x_test
end

function norm(x_train, x_test)
    total_data = vcat(x_train, x_test)
    mach = fit!(machine(Standardizer(), total_data));
    norm_data = MLJ.transform(mach, total_data)
    return norm_data[1:5000,:], norm_data[5001:end,:]
end

function chose_predictors(x_train, x_test, std_)
    stds = std.(eachcol(x_train))
    stds = stds/maximum(stds)
    x_train = x_train[:,stds.>std_]
    x_test = select(x_test, names(x_train))
    stds = std.(eachcol(x_test))
    stds = stds/maximum(stds)
    x_test = x_test[:,stds.>std_]
    x_train = select(x_train, names(x_test))
    return x_train, x_test
end

function clear_const(df)
    df_no_const = df[:,  std.(eachcol(df)) .!= 0]
    return df_no_const
end