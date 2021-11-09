
setwd('C:/Users/josep/OneDrive/Desktop/Y4S1/ZB4171/assignments/Project/spatialVI/')

rm(list=ls())
dev.off()
graphics.off()

library(plyr)
library(pals)
library(plyr)
library(uwot)
library(mclust)
library(dbscan)
library(igraph)
library(ggplot2)
library(schelpers)
library(gridExtra)
library(leidenAlg)
library(data.table)
library(RcppHungarian)


getGraph <- function(x, k) {
    
    snet <- sNN(kNN(x, k = k), k = k)
    from <- shared <- .N <- NULL
    snet.dt = data.table(from = rep(1:nrow(snet$id), k),
                         to = as.vector(snet$id),
                         weight = 1/(1 + as.vector(snet$dist)),
                         distance = as.vector(snet$dist),
                         shared = as.vector(snet$shared))
    data.table::setorder(snet.dt, from, -shared)
    snet.dt[, rank := 1:.N, by = from]
    snet.dt <- snet.dt[rank <= 3 | shared >= 5]
    graph <- graph_from_data_frame(snet.dt)
    return(graph)
    
}

runLeiden <- function(latent, res=0.8, k=20, seed=42) {
    set.seed(seed)
    graph <- getGraph(latent, k=k)
    out <- as.numeric(leiden.community(graph, resolution = res, n.iterations = -1)$membership)
    message('Number of clusters:', max(out))
    return(out)
}

runKmeans <- function(latent, centers, ...) {
    return(kmeans(latent, centers, ...)$cluster)
}

theme_blank <- function(legend.text.size=10,
                        main.size=5,
                        legend.pos='right') {
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          plot.background = element_blank(),
          strip.background = element_blank(),
          plot.title = element_text(size = main.size),
          legend.text = element_text(size = legend.text.size),
          legend.title = element_blank(),
          legend.position = legend.pos,
          legend.key = element_blank())
}

getDiscretePalette <- function(feature, col.discrete) {
    
    num <- is.numeric(feature)
    char <- is.character(feature)
    
    if (num) n <- max(feature)
    if (char) n <- length(unique(feature))
    
    if (is.null(col.discrete)) {
        pal <- getPalette(n)
        if (num) pal <- pal[sort(unique(feature))]
    } else {
        if (length(col.discrete) < n) stop('Not enough colors')
    }
    
    return(pal)
    
}

getPalette <- function(n) {
    all.cols <- c(pals::kelly()[-c(1,3)],
                  pals::glasbey(),
                  pals::polychrome())
    all.cols <- as.character(all.cols)
    return(all.cols[seq_len(n)])
}

runPCA <- function(latent) {
    data <- data.frame(prcomp(latent)$x[,1:2])
    colnames(data) <- c('PC1', 'PC2')
    return(data)
}

runUMAP <- function(latent, n.neighbors=30, spread=2, min.dist=0.1, ...) {
    data <- data.frame(umap(latent, n_neighbors = n.neighbors, 
                            spread = spread, min_dist = min.dist, ...))
    colnames(data) <- c('UMAP1', 'UMAP2')
    return(data)
}

plotMain <- function(coordinates,
                     by = NULL, type = 'discrete',
                     pt.size = 0.5, pt.alpha = 0.9, col.midpoint = NULL, col.low = 'blue',
                     col.mid = 'gray95', col.high = 'red', col.discrete = NULL, main = NULL,
                     main.size = 5, legend = TRUE, legend.text.size = 6, legend.pt.size = 3,
                     n.neighbors = 30, spread = 2, min.dist = 0.1) {
    
    data <- coordinates
    x <- colnames(data)[1]
    y <- colnames(data)[2]
    
    if (is.null(by)) plot <- ggplot(data, aes(x = !!ensym(x), y = !!ensym(y)))
    else {
        
        feature <- by
        data <- cbind(data, feature = feature)
        
        if (type == 'continuous') {
            if (is.null(col.midpoint)) col.midpoint <- median(feature)
            plot <- ggplot(data, aes(x = !!ensym(x), y = !!ensym(y), col = feature)) +
                scale_color_gradient2(midpoint = col.midpoint, low = col.low, 
                                      mid = col.mid, high = col.high)
        }
        
        if (type == 'discrete') {
            plot <- ggplot(data, aes(x = !!ensym(x), y = !!ensym(y), col = as.factor(feature))) +
                scale_color_manual(values = getDiscretePalette(feature, col.discrete)) +
                guides(color = guide_legend(override.aes = list(size = legend.pt.size)))
        }
        
    }
    
    legend.pos <- ifelse(legend, 'right', 'none')
    plot <- plot + geom_point(size = pt.size, alpha = pt.alpha) + ggtitle(main) +
        theme_blank(legend.text.size = legend.text.size,
                    main.size = main.size,
                    legend.pos = legend.pos)
    
    return(plot)
}

mapToSeed <- function(val, seed) {
    con.mat <- table(val, seed)
    cost.mat <- max(con.mat) - con.mat
    matching <- HungarianSolver(cost.mat)$pairs
    
    ## Mapping fr more clusters to fewer
    unmapped <- which(matching[,2] == 0)
    if (is.character(seed)) start <- length(unique(seed))
    if (is.numeric(seed)) start <- max(seed)
    impute <- start + seq_len(length(unmapped))
    matching[,2][matching[,2] == 0] <- impute
    
    new.val <- mapvalues(x = val, from = matching[,1], to = matching[,2])
    return(new.val)
}

##### Run ----------------------------------------------------------------------

all_dirs <- list.dirs('checkpoints', recursive=FALSE, full.names = TRUE)
models <- paste0('CVAE-', seq(10,40,10))

locs <- fread('data/merfish/hypo_ani1_cellcentroids.csv', header=TRUE)
mdata <- fread('data/merfish/hypo_ani1_metadata.csv', header=TRUE)

curr_dirs <- c("checkpoints/CVAE_20211107-201718",
               "checkpoints/CVAE_20211107-044737",
               "checkpoints/CVAE_20211107-221024",
               "checkpoints/CVAE_20211107-200530")

latent <- lapply(curr_dirs, function(direc) {
    fread(list.files(direc, pattern = 'latent', full.names = TRUE), header=TRUE)
})

celltype <- mdata$Cell_class
clusters <- lapply(latent, function(lat) runLeiden(lat, res=0.2))
clusters[[1]] <- mapToSeed(clusters[[1]], clusters[[2]])
clusters[[3]] <- mapToSeed(clusters[[3]], clusters[[2]])
clusters[[4]] <- mapToSeed(clusters[[4]], clusters[[2]])

pca.coords <- lapply(latent, function(lat) runPCA(lat))
umap.coords <- lapply(latent, function(lat) runUMAP(lat))

pca1 <- Map(function(coords, clust, mod) {
    plotMain(coords, by = clust, main = mod, main.size = 18)
}, pca.coords, clusters, models)
pca2 <- lapply(pca.coords, function(coords) plotMain(coords, by = celltype, main = 'Ground truth', main.size = 18))

umap1 <- Map(function(coords, clust, mod) {
    plotMain(coords, by = clust, main = mod, main.size = 18)
}, umap.coords, clusters, models)
umap2 <- lapply(umap.coords, function(coords) plotMain(coords, by = celltype, main = 'Ground truth', main.size = 18))

spatial1 <- Map(function(clust, mod) {
    plotMain(locs, by = clust, main = mod, main.size = 18, pt.size = 1.5)
}, clusters, models)
spatial2 <- plotMain(locs, by = celltype, pt.size = 1, main = 'Ground truth', main.size = 18)
spatialAll <- c(spatial1, list(spatial2))

pdf('analysis/plots/n_neighbors.pdf', height = 20, width = 30)
do.call(grid.arrange, c(pca1, ncol=3))
do.call(grid.arrange, c(pca2, ncol=3))
do.call(grid.arrange, c(umap1, ncol=3))
do.call(grid.arrange, c(umap2, ncol=3))
do.call(grid.arrange, c(spatialAll, ncol=3))
dev.off()
