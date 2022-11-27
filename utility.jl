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
