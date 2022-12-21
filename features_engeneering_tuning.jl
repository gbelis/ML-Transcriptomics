using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse#, Statistics, Distributions,OpenML, MLJFlux, Flux
import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS
# import Pkg; Pkg.add("GLM")
# using GLM

include("./data_processing.jl")
include("./models.jl")

#Importing Data
train_df = load_data("./data/train.csv.gz")
test_df = load_data("./data/test.csv.gz")

Random.seed!(0)

#clean data
x_train, x_test,y = clean_data(train_df, test_df, from_index=true)
#normalisation
#x_train, x_test = norm(x_train, x_test)

x_train = MLJ.transform(fit!(machine(Standardizer(), x_train)), x_train)

train_df = nothing
test_df = nothing
x_test = nothing

############################## Call_rates
"""
Select the predictors with the highest call rates. 
The call rate for a given gene is defined as the proportion of measurement 
for which the corresponding gene information is not 0. 
We try different pourcents of selection and observe the resulting accuracy
"""
# initiate a dataframe to save accuracy values
results = DataFrame([[],[],[]], ["pourcent","length", "accuracy"])
n_folds=5

# pourcents to try
pourcent = collect(0:1:100)

#cross validation using a MultinomialClassifier
for i in (pourcent)
    m = 0.0
    l = 0.0
    println("i:",i)

    #select best features
    indexes_call_rates_CBP = call_rates(x_train[(y.=="CBP"),:], i)
    indexes_call_rates_KAT5 = call_rates(x_train[(y.=="KAT5"),:],i)
    indexes_call_rates_eGFP = call_rates(x_train[(y.=="eGFP"),:],i)
    indexes= unique([indexes_call_rates_CBP; indexes_call_rates_KAT5;indexes_call_rates_eGFP])
    x_train = select(x_train, indexes)
    l = length(x_train[1,:])
    println(l)

    for j in (1:n_folds)
        #cross validation
        t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
        mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
        m += mean(predict_mode(mach, te2) .== tev2)
    end
    push!(results, [i ,l, m/n_folds])
end

results.pourcent = 100 .- results.pourcent

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

results

n_folds=5
Random.seed!(0)

nb_pred = floor.(Int,sort(unique(vcat(range(3000, 21000,length=73), range(200,3000,length=57))), rev=true))

println(results)

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
"""
Select features with both call-rates and mean methods.
"""
# initiate a dataframe to save accuracy values
results_tot = DataFrame(mean=0.0, pourcent=0, length= 9801, accuracy = 0.9)
n_folds=5
Random.seed!(0)
pourcent = collect(22:5:45) # diferent pourcent to try
diff_mean = collect(1000:100:3000) # different mean select numer of predictors

#clean the data

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
"""
Select features with the highest absolute mean
"""

#compute the Mean Absolute Difference
mean_abs_diff =[]
for column in (eachcol(x_train))
    append!(mean_abs_diff, sum(abs.(column .- mean(column))))
end

results_mad= DataFrame(gene = names(x_train), MAD= mean_abs_diff)
sort!(results_mad, [:MAD], rev=true)

# se lect the bests features
selection = results_mad[results_mad.MAD.>100,:]
x_train2 = select(x_train, selection.gene)

# compute the accuracy
train_data, validation_train, test_data, validation_test = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(LogisticClassifier(penalty = :none),train_data, validation_train) |>fit!
m += mean(predict_mode(mach, test_data) .== validation_test)

############################## ftest
"""
Select features according to a f-test
"""

function f_test(x,y)
    """
    Compute the pvalue between two data

    Arguments:
        x {DataFrame column} -- first data
        x {DataFrame column} -- second data

    Returns :
    pValue {float}  -- pvalue
    """
    fStat = var(x) / var(y)
    fDist = FDist(length(x) - 1, length(y) - 1)
    fCdf = cdf(fDist, fStat)
    fCdf < 0.5 ? (pValue = 2 * fCdf) : (pValue = 2 * (1 - fCdf))
    return pValue
end

# initiate a dataframe to save accuracy values
f_test_df = DataFrame([[],[],[], []], ["gene","p_value1", "p_value2", "p_value3"])

#computation of the value of each predictors between eachlabels
for i in range(1,length(x_train[1,:]))
    p1 =f_test(x_train[(y.=="CBP"),i], x_train[(y.=="KAT5"),i])
    p2 =f_test(x_train[(y.=="CBP"),i], x_train[(y.=="eGFP"),i])
    p3 =f_test(x_train[(y.=="KAT5"),i], x_train[(y.=="eGFP"),i])
    push!(f_test_df,[names(x_train)[i],p1,p2,p3] )
end

# select the max pvalue between the 3 previously calculated pvalues
f_test_df.max_pval = max.(f_test_df.p_value1,f_test_df.p_value2,f_test_df.p_value2)

#sort by the best pvalues
sort!(f_test_df, [:max_pval], rev=true)

#select the bests predictors
selection = f_test_df[1:5000,:]
x_train2 = select(x_train, selection.gene)

# compute the accuracy
train_data, validation_train, test_data, validation_test = data_split(x_train2,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(LogisticClassifier(penalty = :none),train_data, validation_train) |>fit!
m += mean(predict_mode(mach, test_data) .== validation_test)v

############# Visualization

res_mean = load_data("./data/results_mean")
res_ttest = load_data("./data/results_ttest")
res_cr = load_data("./data/results_call_rates.csv")

p1 = PlotlyJS.plot(PlotlyJS.scatter(x=res_cr.predictors_nb, y=res_cr.accuracy, mode="markers", marker=attr(size=5, color=res_cr.predictors_nb, colorscale="Viridis", showscale=true)),
Layout(title="1- Accuracy depending on the call rates chosen",yaxis_title="Test accuracy"))

p2 = PlotlyJS.plot(PlotlyJS.scatter(x=res_mean.predictors_nb, y=res_mean.accuracy, mode="markers", marker=attr(size=5, color=res_mean.predictors_nb, colorscale="Viridis", showscale=true)),
Layout(title="2- Predictors selected on the mean variation between labels",yaxis_title="Test accuracy"))

p3 = PlotlyJS.plot(PlotlyJS.scatter(x=res_ttest.predictors_nb, y=res_ttest.accuracy, mode="markers", marker=attr(size=5, color=res_ttest.predictors_nb, colorscale="Viridis", showscale=true)),
Layout(title="3- T-test Predictors selection ",yaxis_title="Test accuracy", xaxis_title="number of predictors", font=attr(size=10)))

p = [p1; p2; p3]

relayout!(p, height=700, width=800)#, title_text="Accuracy variation with different features selection techniques")

p
open("./Features_selection_plot.html", "w") do io PlotlyBase.to_html(io, p.plot) end