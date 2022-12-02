using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, DataFrames, Random, CSV,StatsPlots, MLCourse, Statistics

function pca_cumvar_plot(training_data)
    pca_gene = fit!(machine(PCA(), training_data), verbosity = 0);
    vars = report(pca_gene).principalvars ./ report(pca_gene).tvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
    return report(pca_gene).principalvars
end



#useful tools to interpret data

# @df x_train corrplot([:Xkr4 :Gm19938 :Gm1992 :Rp1],
#                      grid = false, fillcolor = cgrad(), size = (700, 600))

# scatter(pca_df.x1, pca_df.x2, c = y, legend = false)
