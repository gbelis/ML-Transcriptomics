function separate_data_prediction(df)
    return df[:,1:end-1], df[:,end]
end

function clear_const(df)
    df_no_const = df[:,  std.(eachcol(df)) .!= 0]
    return df_no_const
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
function test_80_20(X, Y, n)
    acc_test = 0
    acc_train = 0
    for _ in range(n)
        dat = shuffle(hcat(X,Y))
        X, Y = select(dat, Not(:labels)), dat.labels
        x_test = X[1:1000,:]
        x_train = X[1001:end,:]
        mach = machine(SVC(), x_train, ) |> fit!
        acc_train += mean(predict(mach, x_train) .== y[1001:end,:])
        acc_test = mean(predict(mach, x_test) .== y[1:1000,:])
    end
    return acc_test/n, acc_train/n
end

range(1,5000,1000)