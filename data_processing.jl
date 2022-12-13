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

function call_rates(df,pourcent)
    """
        Return columns with low call rates in a given DataFrame. The call rate for a given gene is defined as the proportion of measurement
        for which the corresponding gene information is not 0. We keep only gene whose call rate is > 1%

    Arguments:
        df {DataFrame} -- data to clean

    Returns :
        df_no_const {DataFrame} -- New DataFrame without proportionnal columns/predictors
    """
    rates = zeros(0)
    for column in (eachcol(df))
        append!( rates,(sum(x->x>0, column) /length(df[:,1])) *100)
    end
    call_rates = DataFrame(index = names(df) , rates = rates )
    call_rates = call_rates[(call_rates.rates.>pourcent),:]
    return call_rates.index
end


function correlation_labels(df,predictors_nb)
    """
        Return columns with low call rates in a given DataFrame. The call rate for a given gene is defined as the proportion of measurement
        for which the corresponding gene information is not 0. We keep only gene whose call rate is > 1%

    Arguments:
        df {DataFrame} -- data to clean

    Returns :
        df_no_const {DataFrame} -- New DataFrame without proportionnal columns/predictors
    """
    mean_CBP = mean.(eachcol(df[(y.=="CBP"),:]))
    mean_KAT5 = mean.(eachcol(df[(y.=="KAT5"),:]))
    mean_eGFP = mean.(eachcol(df[(y.=="eGFP"),:]))
    
    max_diff= max.(abs.(mean_CBP -mean_KAT5),abs.(mean_KAT5-mean_eGFP), abs.(mean_CBP-mean_eGFP))
    results_mean= DataFrame(gene = names(df), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, max_diff=max_diff)
    sort!(results_mean, [:max_diff], rev=true)
   
    selection = results_mean[1:predictors_nb,:]
    x_train = select(df, selection.gene)
    return x_train
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
        path {}
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
        indexes = load_data("./data/indexes_5.csv") #indexes of the cleaned data, to gain time
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
        x_train = remove_prop_predictors(select(x_train, names(x_test)))
        x_test = select(x_test, names(x_train))

        indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], 45)
        indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],45)
        indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],45)
        indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
        x_train = select(x_train, indexes)
        x_test = select(x_test, names(x_train))

        x_train = correlation_labels(x_train,3000)
        x_test = select(x_test, names(x_train))
    
        y = coerce!(train_df, :labels => Multiclass).labels

        if normalised
            x_train, x_test = norm(x_train, x_test)
        end    

        CSV.write("./data/indexes10.csv",DataFrame(index=names(x_train)))
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


function coef_info(beta_df, x_train)
    df = permutedims(beta_df, 1)
    df.stds = std.(eachcol(x_train))
    df.t_value = df[:,2] ./ df.stds
    df.abs_t = abs.(df.t_value)
    return df
end

function get_names(X, y, cutoff)
    mach = machine(MultinomialClassifier(penalty = :none), X, y)
    fit!(mach, verbosity = 0)
    params = fitted_params(mach)
    df = hcat(DataFrame(titles = levels(params.classes)), DataFrame(params.coefs))
    info = DataFrame(genes = names(df[:,2:end]))
    for i in range(1,3,3)
        #info.levels(params.classes)[i] = coef_info(DataFrame(df[i, :]), X)
        info = hcat(info, coef_info(DataFrame(df[Int(i), :]), X).abs_t, makeunique=true)
    end
    info = permutedims(info, 1)
    maxs = DataFrame(genes = names(info[:,2:end]) ,maxs = maximum.(eachcol(info[:,2:end])))
    #chosen_names = names(permutedims(maxs[maxs.maxs .> 1, :], 1)[:,2:end])
    #maxs
    return names(permutedims(maxs[maxs.maxs .> cutoff, :], 1)[:,2:end])
end