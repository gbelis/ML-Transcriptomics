function load_data(path)
    """
        Load the data at "path" localization in the form of a dataframe

    Arguments:
        path {string} -- localization of the data to download

    Returns :
        Dataframe {DataFrame} -- DataFrame with the data
    """
    return DataFrame(CSV.File(path))
end

function remove_constant_predictors(df)
    """
        Remove constant columns in a given DataFrame

    Arguments:
        df {DataFrame} -- data to clean

    Returns :
        df_no_const {DataFrame} -- New DataFrame without constante columns/predictors
    """
    df_no_const = df[:,  std.(eachcol(df)) .!= 0]
    #df_no_const = df[:,  std.(eachcol(df)) .> 0.1]
    return df_no_const
end

function remove_prop_predictors(df)
    """
        Remove exact correlated columns in a given DataFrame

    Arguments:
        df {DataFrame} -- data to clean

    Returns :
        df_no_const {DataFrame} -- New DataFrame without proportionnal columns/predictors
    """
    corr_pairs = findall(≈(1), cor(Matrix(df))) |> idxs -> filter(x -> x[1] > x[2], idxs)
    corr_pred = getindex.(corr_pairs,1)
    corr_pred = unique(corr_pred)
    return df[:,Not(corr_pred)]
end

function norm(x_train, x_test)
    """
        Normalize the data. Compute the norm and apply it to the data with a MLJ transform

    Arguments:
        x_train {DataFrame} -- train set (without labels) to normalize
        x_test {DataFrame} -- test set to normalize

    Returns :
        norm_data[1:5000,:] {DataFrame} -- Normalized train set (x_train)
        norm_data[5001:end,:] {DataFrame} -- Normalized test set (x_test)
    """
    total_data = vcat(x_train, x_test)
    mach = fit!(machine(Standardizer(), total_data));
    norm_data = MLJ.transform(mach, total_data)
    return norm_data[1:5000,:], norm_data[5001:end,:]
end

function clean_data(train_df, test_df; normalised=false, from_index=true)
    """
        Prepare the data by removing constant and correlated predictors. Can also normalized the data.

    Arguments:
        train_df {DataFrame} -- train set (with labels)
        test_df {DataFrame} -- test set
        normalised {Boolean} -- If true the function normalize train and test data.
        from_index {Boolean} -- Determine the way to clean the data. If true, the function read a dataframe
                                containing the indexes to keep. (This was previously done to gain time in the 
                                pre-processing of the data). If false use functions remove_prop_predictors and 
                                remove_constant_predictors to clean the data.

    Returns :
    x_train {DataFrame} -- cleaned train set 
    x_test {DataFrame} -- cleaned test set
    y {DataFrame} -- labels of the training data
    """

    if from_index
        indexes = load_data("./data/indexes.csv") #indexes of the cleaned data, to gain time
        println(size(indexes))
        x_train = select(train_df, indexes.index)
        x_test = select(test_df, indexes.index)
        y = coerce!(train_df, :labels => Multiclass).labels
        if normalised
            x_train, x_test = norm(x_train, x_test)
        end

    else
        x_train = remove_constant_predictors(select(train_df, Not(:labels)))
        x_test = remove_constant_predictors(select(test_df, names(x_train)))
        x_train = select(train_df, names(x_test))

        x_test = remove_prop_predictors(x_test)
        x_train = remove_prop_predictors(select(train_df, names(x_test)))
        x_test = select(test_df, names(x_train))

        y = coerce!(train_df, :labels => Multiclass).labels

        if normalised
            x_train, x_test = norm(x_train, x_test)
        end    

        #CSV.write("./data/indexes3.csv",DataFrame(index=names(x_train)))
    end
    return x_train,x_test,y
end

function kaggle_submit(df_prediction, title)
    """
        Save a csv file for a kaggle submission with prediction for the test set

    Arguments:
        df_prediction {DataFrame} -- prediction for the test set
        title {string} -- name of the file to save

    """

    prediction_kaggle = DataFrame(id = collect(1:length(df_prediction)))
    prediction_kaggle[!,:prediction] = df_prediction
    CSV.write("./Submission/$(title).csv", prediction_kaggle)
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


function pca(df,dimension)
    """
        Do a pca 

    Arguments:
        df {DataFrame} -- data on which to do the pca
        dimension {int} -- dimension of the pca

    Returns :
        df_no_const {DataFrame} -- New DataFrame without proportionnal columns/predictors
    """
    return MLJ.transform(fit!(machine(PCA(maxoutdim = dimension), df)), df)
end


function clean(train_df, test_df)
    x_train = remove_constant_predictors(select(train_df, Not(:labels)))
    x_test = remove_constant_predictors(select(test_df, names(x_train)))
    x_train = select(train_df, names(x_test))
    return x_train, x_test
end

function no_corr(x_train, x_test)
    x_test = remove_prop_predictors(x_test)
    x_train = remove_prop_predictors(select(x_train, names(x_test)))
    x_test = select(test_df, names(x_train))
    return x_train, x_test
end