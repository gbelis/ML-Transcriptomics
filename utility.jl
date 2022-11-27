function separate_data_prediction(df)
    return df[:,1:end-1], df[:,end]
end

function clean_data(df)
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
    CSV.write("./Submission/title.txt", prediction_kaggle)
end