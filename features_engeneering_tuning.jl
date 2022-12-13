using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV, StatsPlots, MLJ, MLJLinearModels, MLCourse, Statistics, Distributions,OpenML,  MLJMultivariateStatsInterface, NearestNeighborModels, MLJFlux, Flux, LinearAlgebra
# import Pkg; Pkg.add("PlotlyJS")
# using PlotlyJS
# import Pkg; Pkg.add("GLM")
# using GLM

include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

Random.seed!(0)


x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
# data=vcat(x_train,x_test)
foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(x_train))
#x_train= data[1:5000,:]

mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))#./ std.(eachcol(x_train[(y.=="CBP"),:]))
#mean_CBP2 = sum.(eachcol(x_train[(y.=="CBP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="CBP"),:]))
mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
#mean_KAT52 = sum.(eachcol(x_train[(y.=="KAT5"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="KAT5"),:]))
mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
#mean_eGFP2 = sum.(eachcol(x_train[(y.=="eGFP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="eGFP"),:]))

max_diff= max.(abs.(mean_CBP -mean_KAT5),abs.(mean_KAT5-mean_eGFP), abs.(mean_CBP-mean_eGFP))

results_mean= DataFrame(gene = names(x_train), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, max_diff=max_diff)
sort!(results_mean, [:max_diff], rev=true)

selection = results_mean[1:500,:]
x_train2 = select(x_train, selection.gene)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
m = mean(predict_mode(mach, te2) .== tev2)
CSV.write("./data/indexes_500.csv", DataFrame(index=names(x_train)))


indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], 60)
indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],60)
indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],60)
indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
x_train2 = select(x_train, indexes)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
m = mean(predict_mode(mach, te2) .== tev2)

CSV.write("./data/indexes_new.csv", DataFrame(index=names(x_train2)))
sum.(eachrow(x_train))


############################## Call_rates
results = DataFrame(pourcent= 0., length= 9800,accuracy = 0.9)
n_folds=5

pourcent = collect(0:1:50)
println(pourcent)

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
        mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

CSV.write("./data/results_call_rates.csv",results)

PlotlyJS.plot(PlotlyJS.scatter(x=results.pourcent, y=results.accuracy, mode="markers", marker=attr(size=8, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the call rates chosen",yaxis_title="Test accuracy", xaxis_title="Pourcent of call rates removed", coloraxis_title = "number of predictors"))


############################## Supervised Mean Difference
results = DataFrame(mean=0.0, length= 9801, accuracy = 0.9)
n_folds=5
Random.seed!(0)

x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(x_train))

mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))#./ std.(eachcol(x_train[(y.=="CBP"),:]))
#mean_CBP2 = sum.(eachcol(x_train[(y.=="CBP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="CBP"),:]))
mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
#mean_KAT52 = sum.(eachcol(x_train[(y.=="KAT5"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="KAT5"),:]))
mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
#mean_eGFP2 = sum.(eachcol(x_train[(y.=="eGFP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="eGFP"),:]))

max_diff= max.(abs.(mean_CBP -mean_KAT5),abs.(mean_KAT5-mean_eGFP), abs.(mean_CBP-mean_eGFP))
results_mean= DataFrame(gene = names(x_train), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, max_diff=max_diff)
sort!(results_mean, [:max_diff], rev=true)

diff_label = collect(0:0.005:0.2)

for i in (diff_label)
    m = 0.0
    l = 0.0
    selection = results_mean[results_mean.max_diff.>i,:]
    x_train2 = select(x_train, selection.gene)
    l = length(x_train2[1,:])
    println(l)

    for j in (1:n_folds)
        t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(LogisticClassifier(penalty = :none),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

CSV.write("./data/results_mean.csv",results)

PlotlyJS.plot(PlotlyJS.scatter(x=results.mean, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the variability between labels",yaxis_title="Test accuracy", xaxis_title="difference between the means of labels", coloraxis_title = "number of predictors"))

############################## Mean and Call_rates
results_tot = DataFrame(mean=0.0, pourcent=0, length= 9801, accuracy = 0.9)
n_folds=5
Random.seed!(0)
pourcent = collect(22:5:45)
diff_mean = collect(0.058:0.01:0.14)


x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)

for i in (pourcent)

    x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
    indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], i)
    indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],i)
    indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],i)
    indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
    x_train = select(x_train, indexes)
    x_test = select(x_test, names(x_train))
   
    for j in (diff_mean)
        println("pourcent $(i), cutoff $(j)")
        m = 0.0
        l=0.0
        mean_CBP = mean.(eachcol(x_train[(y.=="CBP"),:]))#./ std.(eachcol(x_train[(y.=="CBP"),:]))
        #mean_CBP2 = sum.(eachcol(x_train[(y.=="CBP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="CBP"),:]))
        mean_KAT5 = mean.(eachcol(x_train[(y.=="KAT5"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
        #mean_KAT52 = sum.(eachcol(x_train[(y.=="KAT5"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="KAT5"),:]))
        mean_eGFP = mean.(eachcol(x_train[(y.=="eGFP"),:]))#./ std.(eachcol(x_train[(y.=="KAT5"),:]))
        #mean_eGFP2 = sum.(eachcol(x_train[(y.=="eGFP"),:]))./sum.(x->x>0, eachcol(x_train[(y.=="eGFP"),:]))

        max_diff= max.(abs.(mean_CBP -mean_KAT5),abs.(mean_KAT5-mean_eGFP), abs.(mean_CBP-mean_eGFP))
        results_mean= DataFrame(gene = names(x_train), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, max_diff=max_diff)
        sort!(results_mean, [:max_diff], rev=true)
        selection = results_mean[results_mean.max_diff.>j,:]
        x_train2 = select(x_train, selection.gene)
        l = length(x_train2[1,:])
        println(l)

        for k in (1:n_folds)
            train_data, validation_train, test_data, validation_test = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
            mach = machine(LogisticClassifier(penalty = :none),train_data, validation_train) |>fit!
            m += mean(predict_mode(mach, test_data) .== validation_test)
        end
        push!(results_tot, [j,i,l, m/n_folds])
    end
end

println(results_tot)
sort!(results_tot, [:length], rev=true)


CSV.write("./data/results_tot.csv",results)
PlotlyJS.plot(PlotlyJS.scatter(x=results.mean, y=results.pourcent, mode="markers", marker=attr(size=5, color=results.accuracy, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the variability between labels",yaxis_title="Test accuracy", xaxis_title="difference between the means of labels", coloraxis_title = "number of predictors"))


############################## Mean Absolute Difference (MAD)
mean_abs_diff =[]
for column in (eachcol(x_train))
    append!(mean_abs_diff, sum(abs.(column.-mean(column))))
end
mean_abs_diff

results_mad= DataFrame(gene = names(x_train), MAD= mean_abs_diff)
sort!(results_mad, [:MAD], rev=true)

selection = results_mad[results_mad.MAD.>100,:]
x_train2 = select(x_train, selection.gene)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(LogisticClassifier(penalty = :none),t2, tv2) |>fit!
m = mean(predict_mode(mach, te2) .== tev2)


############################## f-test ANOVA

fs_scores = load_data("./data/fs_scores.csv")
rename!(fs_scores,2 => :scores)
rename!(fs_scores,1 => :predictors)
fs_scores.predictors= fs_scores.predictors.+1

score_threshold= collect(0:5:50)

for i in (score_threshold)
    m = 0.0
    l = 0.0
    fs_scores = fs_scores[fs_scores.scores.>i,:]
    predictors = floor.(Int,Matrix(fs_scores)[:,1])
    x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
    x_train = x_train[:,predictors]
    l = length(x_train[1,:])
    println(l)

    for j in (1:n_folds)
        t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(LogisticClassifier(penalty = :none),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

CSV.write("./data/results3.csv",results)
PlotlyJS.plot(PlotlyJS.scatter(x=results.f_threshold, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the f-test score threshold chosen",yaxis_title="Test accuracy", xaxis_title="threshold of f-test", coloraxis_title = "number of predictors"))

############################## t-test and call_rates

pourcent = collect(0:5:30)
t_cutoff = collect(0:0.5:3)

for i in (pourcent)

    x_train,x_test,y = clean_data(train_df, test_df, normalised=false, from_index=true)
    indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], i)
    indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],i)
    indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],i)
    indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
    x_train = select(x_train, indexes)

    for j in (t_cutoff)
        m = 0.0
        l=0.0

        for k in (1:n_folds)
            println("pourcent $(i), cutoff $(j), fold $(k)")
            train_x, train_y, test_x, test_y = data_split(x_train,y, 1:4000, 4001:5000)
            pred_names = get_names(train_x, train_y, j)
            train_x = select(train_x, pred_names)
            l = length(train_x[1,:])
            println(l)
            mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y)
            fit!(mach, verbosity = 0)
            m += mean(predict_mode(mach, select(test_x, pred_names)) .== test_y)
        end
        push!(results, [j,i,l, m/n_folds])
    end
end

println(results)
CSV.write("./data/results2.csv",results)
PlotlyJS.plot(PlotlyJS.scatter(x=results.pourcent, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the call rates chosen",yaxis_title="Test accuracy", xaxis_title="Pourcent of call rates removed", coloraxis_title = "number of predictors"))


#functions
function coef_info(beta_df, x_train)
    df = permutedims(beta_df, 1)
    df.stds = std.(eachcol(x_train))
    df.t_value = df[:,2] ./ df.stds
    df.abs_t = abs.(df.t_value)
    return df
end

function get_names(X, y, cutoff)
    Random.seed!(0)
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

############################## ftest

function f_test(x,y)
    fStat = var(x) / var(y)
    fDist = FDist(length(x) - 1, length(y) - 1)
    fCdf = cdf(fDist, fStat)
    fCdf < 0.5 ? (pValue = 2 * fCdf) : (pValue = 2 * (1 - fCdf))
    return pValue
end

f_test_df = DataFrame([[],[],[], []], ["gene","p_value1", "p_value2", "p_value3"])

for i in range(1,length(x_train[1,:]))
    p1 =f_test(x_train[(y.=="CBP"),i], x_train[(y.=="KAT5"),i])
    p2 =f_test(x_train[(y.=="CBP"),i], x_train[(y.=="eGFP"),i])
    p3 =f_test(x_train[(y.=="KAT5"),i], x_train[(y.=="eGFP"),i])
    push!(f_test_df,[names(x_train)[i],p1,p2,p3] )
end

f_test_df.max_pval = max.(f_test_df.p_value1,f_test_df.p_value2,f_test_df.p_value2)

f_test_df
sort!(f_test_df, [:max_pval], rev=true)

selection = f_test_df[1:5000,:]
x_train2 = select(x_train, selection.gene)

t2,tv2,te2,tev2 = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
m = mean(predict_mode(mach, te2) .== tev2)