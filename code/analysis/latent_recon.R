
##### Libs and functions -------------------------------------------------------

setwd('C:/Users/josep/OneDrive/Desktop/Y4S1/ZB4171/assignments/Project/spatialVI/')

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

runLeiden <- function(latent, res=0.8, k=20) {
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
all_dirs
curr_dir <- all_dirs[1]
model <- 'CVAE'
locs <- fread('data/merfish/hypo_ani1_cellcentroids.csv', header=TRUE)
mdata <- fread('data/merfish/hypo_ani1_metadata.csv', header=TRUE)

## Explore latent space

latent <- fread(list.files(curr_dir, pattern = 'latent', full.names = TRUE), header=TRUE)

celltype <- mdata$Cell_class
clusters <- runLeiden(latent, res=0.2)

# clusters <- mapToSeed(clusters, celltype)

pca.coords <- runPCA(latent)
umap.coords <- runUMAP(latent)

pca1 <- plotMain(pca.coords, by = clusters)
pca2 <- plotMain(pca.coords, by = celltype)

pdf(paste0('analysis/plots/pca-', model, '.pdf'), height = 6, width = 12)
grid.arrange(pca1, pca2, ncol = 2)
dev.off()

umap1 <- plotMain(umap.coords, by = clusters)
umap2 <- plotMain(umap.coords, by = celltype)

pdf(paste0('analysis/plots/umap-', model, '.pdf'), height = 6, width = 12)
grid.arrange(umap1, umap2, ncol = 2)
dev.off()

spatial1 <- plotMain(locs, by = clusters)
spatial2 <- plotMain(locs, by = celltype)

pdf(paste0('analysis/plots/spatial-', model, '.pdf'), height = 6, width = 12)
grid.arrange(spatial1, spatial2, ncol = 2)
dev.off()

pdf(paste0('analysis/plots/spatial-wrap-', model, '.pdf'), height = 10, width = 20)
grid.arrange(spatial1 + facet_wrap(~ feature), 
             spatial2 + facet_wrap(~ feature), 
             ncol = 2)
dev.off()

## Explore reconstruction

## CELL-CELL correlation

counts <- fread('data/merfish/hypo_ani1_counts.csv')
recon <- fread(list.files(curr_dir, pattern = 'recon', full.names = TRUE), header=TRUE)
colnames(recon) <- colnames(counts)

# Subset to celltype
table(celltype)
type <- 'OD'
counts <- counts[!which(grepl(type, celltype))]
recon <- recon[!which(grepl(type, celltype))]

p.cor <- sapply(1:nrow(counts), function(i) cor(as.numeric(counts[i,]), as.numeric(recon[i,]), method = 'pearson'))

# pdf(paste0('analysis/plots/recon-pearson-', model, '.pdf'), height = 5, width = 6)
hist(p.cor, freq = FALSE, breaks=100, col = rgb(0.5,0.5,0.9,0.5), main = '', xlab = '', border = FALSE, xlim = c(0,1))
abline(v=mean(p.cor), col = 'red', lty = 3)
# dev.off()

which.max(p.cor)
count_best <- as.numeric(counts[which.max(p.cor)])
recon_best <- as.numeric(recon[which.max(p.cor)])
kid <- count_best > 0 & recon_best > 0
count_best <- count_best[kid]
recon_best <- recon_best[kid]
plot(log(1+count_best),log(1+recon_best), pch = 19)

which.min(p.cor)
count_worst <- as.numeric(counts[which.min(p.cor)])
recon_worst <- as.numeric(recon[which.min(p.cor)])
kid <- count_worst > 0 & recon_worst > 0
count_worst <- count_worst[kid]
recon_worst <- recon_worst[kid]
plot(log(1+count_worst),log(1+recon_worst), pch = 19)

## Explore ELBO landscape for different models

library(reticulate)
pd <- import('pandas')
nb_train_elbo <- pd$read_pickle(list.files('checkpoints/NBVAE/', pattern = 'train_elbo', full.names=TRUE))
zinb_train_elbo <- pd$read_pickle(list.files('checkpoints/ZINBVAE_20211106-230443/', pattern = 'train_elbo', full.names=TRUE))

nb_train_df <- data.frame(train_elbo = nb_train_elbo, epochs = seq_len(length(nb_train_elbo)), type = 'NB')
zinb_train_df <- data.frame(train_elbo = zinb_train_elbo, epochs = seq_len(length(zinb_train_elbo)), type = 'ZINB') 
train_df <- rbind(nb_train_df, zinb_train_df)

ggplot(train_df, aes(x = epochs, y = train_elbo, color = type)) + 
    geom_line(alpha = 1) +
    xlab('Epochs') + ylab('Training ELBO') + theme_blank()

nb_train_elbo <- pd$read_pickle(list.files('checkpoints/NBVAE/', pattern = 'val_elbo', full.names=TRUE))
zinb_train_elbo <- pd$read_pickle(list.files('checkpoints/ZINBVAE_20211106-230443/', pattern = 'val_elbo', full.names=TRUE))

nb_train_df <- data.frame(train_elbo = nb_train_elbo, epochs = seq_len(length(nb_train_elbo)), type = 'NB')
zinb_train_df <- data.frame(train_elbo = zinb_train_elbo, epochs = seq_len(length(zinb_train_elbo)), type = 'ZINB') 
train_df <- rbind(nb_train_df, zinb_train_df)

ggplot(train_df, aes(x = epochs, y = train_elbo, color = type)) + 
    geom_line(alpha = 1) +
    xlab('Epochs') + ylab('Validation ELBO') + theme_blank()





