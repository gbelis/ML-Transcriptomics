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
x_train = load_data("./data/x_train_norm.csv.gz")
y = coerce!(train_df, :labels => Multiclass).labels
train_df = nothing

#clean data
#x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#standardize data
#x_train, x_test = norm(x_train, x_test)

# pred_names = get_names_len(x_train, y, 6000)
# x_train = select(x_train, pred_names)

indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], 25)
indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],25)
indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],25)
indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
x_train2 = select(x_train, indexes)
indexes_call_rates_CBP =nothing
indexes_call_rates_KAT5 = nothing
indexes_call_rates_eGFP = nothing
indexes=nothing


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

###################################################### Tuning

x_train

model2 = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 128,
                σ = relu),
                optimiser = ADAM(),
                batch_size = 256,
                epochs = 1000)

model = NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(3000, 100, relu), Dense(100, 3), dropout =0.5)),
                                optimiser = ADAM(),
                                batch_size = 256,
                                epochs = 1000)

tuned_model2 = TunedModel(model = model2,
                        resampling = CV(nfolds = 3),
                        tuning = Grid(goal =4),
                        range = #range(model2, :(builder.dropout), values = [0.,.25,.5,1.]),
                                range(model2, :alpha, values = [0.,0.25,0.5,1.]),
                                #range(model2, :epochs, values = [100,1000]),
                        measure = MisclassificationRate())

mach2 = fit!(machine(tuned_model2,x_train2, y), verbosity = 1)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach2 = fit!(machine(model2,t2, tv2), verbosity = 1)
m = mean(predict_mode(mach2, te2) .== tev2)

report(mach2)
plot(mach2)
fitted_params(mach2).best_model

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
