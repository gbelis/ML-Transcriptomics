using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
include("../data_processing.jl")
include("../models.jl")

"""
Neural Network Classification using cross-validation. Save the prediction in a csv file.

x_train {DataFrame} -- train set (without labels)
x_test {DataFrame} -- test set to predict
y {DataFrame} -- labels of the training data
Hyperparameters tuned :

"""


#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#standardize data
x_train, x_test = norm(x_train, x_test)

mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))
mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))
mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))

results_mean= DataFrame(gene = names(x_train), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, diff1=abs.(mean_CBP-mean_eGFP), diff2=abs.(mean_eGFP -mean_KAT5), diff3=(abs.(mean_CBP -mean_KAT5)))

sort!(results_mean, [:diff1], rev=true)
selection1 = results_mean[1:6000,:] 
sort!(results_mean, [:diff2], rev=true)
selection2 = results_mean[1:6000,:]
sort!(results_mean, [:diff3], rev=true)
selection3 = results_mean[1:6000,:]

x_train2 = select(x_train, unique([selection1.gene; selection2.gene; selection3.gene]))

x_train2 = MLJ.transform(fit!(machine(PCA(maxoutdim = 3000), x_train2)), x_train2)

Random.seed!(0)

###################################################### Tuning dropout

model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu),
                optimiser = ADAM(),
                batch_size = 128,
                epochs = 1000)

tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 5),
                        tuning = Grid(goal =4),
                        range = range(model, :(builder.dropout), values = [0.,.25,.5,1.]),
                        measure = MisclassificationRate())

mach = fit!(machine(tuned_model,x_train2, y), verbosity = 1)

report(mach)
plot(mach)
fitted_params(mach).best_model

###################################################### alpha tuning

model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout =0.5),
                optimiser = ADAM(),
                batch_size = 128,
                epochs = 1000)

tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 5),
                        tuning = Grid(goal =4),
                        range = range(model, :alpha, values = [0.,0.25,0.5,1.]),
                        measure = MisclassificationRate())

mach = fit!(machine(tuned_model,x_train2, y), verbosity = 1)

report(mach)
plot(mach)
fitted_params(mach).best_model

###################################################### Epochs Tuning

model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout =0.5),
                optimiser = ADAM(),
                batch_size = 128,
                alpha =0.25)

tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 5),
                        tuning = Grid(goal =4),
                        range = range(model, :epochs, values = [800,1000,1200,2000]),
                        measure = MisclassificationRate())

mach = fit!(machine(tuned_model,x_train2, y), verbosity = 1)

report(mach)
plot(mach)
fitted_params(mach).best_model

###################################################### Batch size Tuning

model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout =0.5),
                optimiser = ADAM(),
                epochs = 1000,
                alpha =0.25)

tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 5),
                        tuning = Grid(goal =3),
                        range = range(model, :batch_size, values = [64,128,256]),
                        measure = MisclassificationRate())

mach = fit!(machine(tuned_model,x_train2, y), verbosity = 1)
report(mach)
plot(mach)
fitted_params(mach).best_model

###################################################### n_hidden neurons tuning

n_neuron = [128, 300]
n_folds = 2
results = DataFrame([[],[]], ["n_neuron", "accuracy"])

x_train2

for i in n_neuron
        m =0.0
        model2 = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = i,
                        σ = relu, dropout = 0.5),
                        optimiser = ADAM(),
                        batch_size = 128,
                        epochs = 2000,
                        alpha = 0.25)

        for k in (1:n_folds)
                train_data, validation_train, test_data, validation_test = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
                mach2 = fit!(machine(model2,train_data, validation_train), verbosity = 1)
                m += mean(predict_mode(mach2, test_data) .== validation_test)
        end
        push!(results, [i , m/n_folds])
    end

###################################################### Best Model

model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout = 0.5),
                optimiser = ADAM(),
                batch_size = 128,
                epochs = 2000,
                alpha = 0.25)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach2 = fit!(machine(model,t2, tv2), verbosity = 1)
m = mean(predict_mode(mach2, te2) .== tev2)









model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu, dropout =0.5),
                optimiser = ADAM(),
                batch_size = 128,
                alpha =0.25)

tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 5),
                        tuning = Grid(goal =2),
                        range = range(model, :epochs, values = [1200,2000]),
                        measure = MisclassificationRate())

mach = fit!(machine(tuned_model,x_train2, y), verbosity = 1)

report(mach)
plot(mach)
fitted_params(mach).best_model
pred = predict_mode(mach, x_test)
kaggle_submit(pred, "NN_mean")





model2 = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(3000, 128, relu), Dense(100, 3), dropout =0.5)),
    optimiser = ADAM(),
    batch_size = 128,
    alpha =0.25)

tuned_model2 = TunedModel(model = model2,
    resampling = CV(nfolds = 5),
    tuning = Grid(goal =2),
    range = range(model, :epochs, values = [1200,2000]),
    measure = MisclassificationRate())

mach2 = fit!(machine(tuned_model2,x_train2, y), verbosity = 1)

report(mach2)
plot(mach2)
fitted_params(mach2).best_model
pred = predict_mode(mach2, x_test)
kaggle_submit(pred, "NN_chain_mean")
