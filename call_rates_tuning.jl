using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux
import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS
include("./data_processing.jl")
include("./data_analysis.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

results = DataFrame(pourcent= 0., length= 24101,accuracy = 0.90938)
n_folds=5

pourcent = sort(unique(vcat(range(0,30,length=31),
                 range(35,100,length=14))))

for i in (pourcent)
    m = 0.0
    l = 0.0
    println("i:",i)

    x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
    indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], i)
    indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],i)
    indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],i)
    indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
    x_train = select(x_train, indexes)
    x_test = select(x_test, names(x_train))
    l = length(x_test[1,:])
    println(l)

    for j in (1:n_folds)
        t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:l1, lambda = 7.83e-5),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)
CSV.write("./data/results.csv",results)
PlotlyJS.plot(PlotlyJS.scatter(x=results.pourcent, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the call rates chosen",yaxis_title="Test accuracy", xaxis_title="Pourcent of call rates removed", coloraxis_title = "number of predictors"))

