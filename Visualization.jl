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

# Histogram of distribution of label in the training set
df = DataFrame( label = ["CBP","KAT5","eGFP"], 
                sample_nb = [length(x_train[(y.=="CBP"),:][:,1]), 
                            length(x_train[(y.=="KAT5"),:][:,1]), 
                            length(x_train[(y.=="eGFP"),:][:,1])])

PlotlyJS.plot(histogram(histfunc="count", x= ["CBP", "KAT5", "eGFP"], y=y, name="count"))

p1 = plot(df, x=:label, y=:sample_nb, height=300, kind="bar", Layout(title="Distributions of experimental conditions in the training data"))
open("./histogram.html", "w") do io PlotlyBase.to_html(io, p1.plot) end

#Confusion matrix
t2,tv2,te2,tev2 = data_split(x_train,y, 1:4000, 4001:5000, shuffle =true)
mach = machine(MultinomialClassifier(penalty =:none),t2, tv2) |>fit!
m = mean(predict_mode(mach, te2) .== tev2)
confusion_matrix(predict_mode(mach, te2), tev2)



