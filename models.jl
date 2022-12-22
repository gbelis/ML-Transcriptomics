# This file contains functions for evaluation and cross validation of machine learning techniques

function evaluate(mach, training_data, validation_data, test_data,validation_test)
    """
        Evaluate the data

    Arguments:
        training_data {DataFrame} -- training data
        valididation_data {DataFrame} -- training labels
        test_data {DataFrame} -- test data
        validation_test {DataFrame} -- test label

    Returns:
        error {DataFrame} -- dataframe of training and test error

    """
    error = DataFrame(trainin_error = mean(predict_mode(mach, training_data) .!= validation_data), test_error = mean(predict_mode(mach, test_data) .!= validation_test))
    return error
end

function data_split(data,y, idx_train, idx_test; shuffle = true)
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