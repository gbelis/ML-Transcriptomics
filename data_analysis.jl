using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV,StatsPlots, MLCourse, Statistics

#useful tools to interpret data

@df x_train corrplot([:Xkr4 :Gm19938 :Gm1992 :Rp1],
                     grid = false, fillcolor = cgrad(), size = (700, 600))

scatter(pca_df.x1, pca_df.x2, c = y, legend = false)
