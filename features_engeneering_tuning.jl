using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse#, Statistics, Distributions,OpenML, MLJFlux, Flux
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

#clean data
x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
#normalisation
x_train, x_test = norm(x_train, x_test)

############################## Call_rates
"""
Select the predictors with the highest call rates. 
The call rate for a given gene is defined as the proportion of measurement 
for which the corresponding gene information is not 0. 
We try different pourcents of selection and observe the resulting accuracy
"""
# initiate a dataframe to save accuracy values
results = DataFrame(pourcent= 0., length= 9800,accuracy = 0.9)
n_folds=5

# pourcents to try
pourcent = collect(0:1:50)
println(pourcent)

#cross validation using a MultinomialClassifier
for i in (pourcent)
    m = 0.0
    l = 0.0
    println("i:",i)

    #select best features
    x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
    indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], i)
    indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],i)
    indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],i)
    indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
    x_train = select(x_train, indexes)
    x_test = select(x_test, names(x_train))
    l = length(x_test[1,:])
    println(l)

    for j in (1:n_folds)
        #cross validation
        t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

println(results)

#save the results
CSV.write("./data/results_call_rates.csv",results)

#make a plot of the accuracy depending of the pourcent of call rates chosen
PlotlyJS.plot(PlotlyJS.scatter(x=results.pourcent, y=results.accuracy, mode="markers", marker=attr(size=8, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="accuracy depending on the call rates chosen",yaxis_title="Test accuracy", xaxis_title="Pourcent of call rates removed", coloraxis_title = "number of predictors"))

############################## Mean Difference
"""
Select the predictors with the highest Mean difference between the labels. 
We compute the difference of means between the three labels, and select a 
certain number of the bests predictors for each 2 predictors. THis allow to 
keep predictors that has a difference of value depending of one and another
label type. 
We try different number of predictors of and observe the resulting accuracy
"""

# initiate a dataframe to save accuracy values
results = DataFrame([[],[],[]], ["length","predictors_nb", "accuracy"])
n_folds=5
Random.seed!(0)

nb_pred = floor.(Int,sort(unique(vcat(range(3000, 21000,length=73),
                                    range(200,3000,length=57))), rev=true))

nb_pred = collect(2000:500:10000)

for i in (nb_pred)
    println("i ", i)
    m = 0.0
    l = 0.0
    for j in (1:n_folds)
        #select best features
        train_data, validation_train, test_data, validation_test = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mean_CBP = mean.(eachcol(train_data[(validation_train.=="CBP"),:]))
        mean_KAT5 = mean.(eachcol(train_data[(validation_train.=="KAT5"),:]))
        mean_eGFP = mean.(eachcol(train_data[(validation_train.=="eGFP"),:]))
        results_mean= DataFrame(gene = names(train_data), CBP= mean_CBP, KAT5= mean_KAT5, eGFP = mean_eGFP, diff1=abs.(mean_CBP-mean_eGFP), diff2=abs.(mean_eGFP -mean_KAT5), diff3=(abs.(mean_CBP -mean_KAT5)))
    
        sort!(results_mean, [:diff1], rev=true)
        selection1 = results_mean[1:i,:] 
        sort!(results_mean, [:diff2], rev=true)
        selection2 = results_mean[1:i,:]
        sort!(results_mean, [:diff3], rev=true)
        selection3 = results_mean[1:i,:]
    
        x_train2 = select(train_data, unique([selection1.gene; selection2.gene; selection3.gene]))
        test_data = select(test_data, names(x_train2))

        l += length(x_train2[1,:])
        println(i, " length : ", l)

        #cross validation
        mach = machine(LogisticClassifier(penalty = :none),x_train2, validation_train) |>fit!
        m += mean(predict_mode(mach, test_data) .== validation_test)
    end
    push!(results, [i ,l/n_folds, m/n_folds])
end

#save the results
CSV.write("./data/results_mean3.csv",results)

#make a plot of the accuracy depending of the pourcent of call rates chosen
PlotlyJS.plot(PlotlyJS.scatter(x=results.predictors_nb, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="Predictors selected on the variability between labels",yaxis_title="Test accuracy", xaxis_title="predictors_nb", coloraxis_title = "number of predictors"))

############################## Mean and Call_rates
results_tot = DataFrame(mean=0.0, pourcent=0, length= 9801, accuracy = 0.9)
n_folds=5
Random.seed!(0)
pourcent = collect(22:5:45)
diff_mean = collect(0.058:0.01:0.14)


x_train,x_test,y = clean_data(train_df, test_df, from_index=true)

for i in (pourcent)

    x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
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
    x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
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

    x_train,x_test,y = clean_data(train_df, test_df, from_index=true)
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

train_x, train_y, test_x, test_y = data_split(x_train,y, 1:4000, 4001:5000)
pred_names = get_names(train_x, train_y, 1.5)
train_x = select(train_x, pred_names)
l = length(train_x[1,:])
println(l)

mach = machine(MultinomialClassifier(penalty = :none), train_x, train_y)
fit!(mach, verbosity = 0)
m = mean(predict_mode(mach, test_x) .== test_y)





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

######### Visualization

res_mean = load_data("./data/results_mean4.csv")
res_ttest = load_data("./data/results_ttest.csv")
res_cr = load_data("./data/results_call_rates.csv")

p1 = PlotlyJS.plot(PlotlyJS.scatter(x=results.pourcent, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="Accuracy depending on the call rates chosen",yaxis_title="Test accuracy", xaxis_title="Pourcent of call rates removed", coloraxis_title = "number of predictors"))

p2 = PlotlyJS.plot(PlotlyJS.scatter(x=results.predictors_nb, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="Predictors selected on the variability between labels",yaxis_title="Test accuracy", xaxis_title="predictors_nb", coloraxis_title = "number of predictors"))

p3 = PlotlyJS.plot(PlotlyJS.scatter(x=results.predictors_nb, y=results.accuracy, mode="markers", marker=attr(size=5, color=results.length, colorscale="Viridis", showscale=true)),
Layout(title="T-test Predictors selection ",yaxis_title="Test accuracy", xaxis_title="predictors_nb", coloraxis_title = "number of predictors"))

p = [p1; p2; p3]

relayout!(p, height=600, width=600, title_text="Accuracy variation with different features selection techniques")

p
open("../Plots/Features_selection_plot.html", "w") do io PlotlyBase.to_html(io, p.plot)