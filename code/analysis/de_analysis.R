

##### Libs and functions -------------------------------------------------------

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

plotScatter <- function (x,y, annotate = NULL, 
                         cor.method = "pearson", log = TRUE, pseudocount = 1, 
                         contour = TRUE, density = FALSE, diagonal = TRUE, pt.alpha = 0.5, 
                         pt.size = 3, contour.alpha = 0.5, contour.size = 0.7, density.alpha = 0.5) 
{
    nna <- !is.na(x) & !is.na(y) & x > 0 & y > 0
    x <- x[nna]
    y <- y[nna]
    if (log) {
        x <- log2(x + pseudocount)
        y <- log2(y + pseudocount)
    }
    corIJ <- cor(x, y, method = cor.method, use = "complete.obs")
    max.lim <- max(max(x), max(y))
    data <- data.frame(x = x, y = y)
    scatter <- ggplot(data = data, aes_string(x = "x", y = "y", size = annotate)) + 
        geom_point(alpha = pt.alpha) + 
        theme_minimal() + theme(legend.position = "right") + 
        xlim(0, max.lim) + 
        ylim(0, max.lim) + labs(title = paste0("CC = ", round(corIJ, 3)))
    
    if (contour) 
        scatter <- scatter + geom_density_2d(alpha = contour.alpha, 
                                             size = contour.size, 
                                             aes(color = ..level..)) + 
        scale_color_viridis_c() + guides(color = "none")
    if (density) 
        scatter <- scatter + geom_density_2d_filled(alpha = density.alpha) + 
        guides(fill = "none")
    if (diagonal) 
        scatter <- scatter + geom_abline(intercept = 0, slope = 1, 
                                         color = 2, alpha = 0.5, linetype = "dashed")
    return(scatter)
}

colpal <- function(i) {
    c <- getPalette(20)
    c[-c(i)] <- 'grey'
    c <- sort(unique(c))
    return(c)
}

##### Run ----------------------------------------------------------------------

all_dirs <- list.dirs('checkpoints', recursive=FALSE, full.names = TRUE); all_dirs
curr_dir <- all_dirs[2]; curr_dir
model <- 'CVAE-40'

locs <- fread('data/merfish/hypo_ani1_cellcentroids.csv', header=TRUE)
mdata <- fread('data/merfish/hypo_ani1_metadata.csv', header=TRUE)
celltype <- mdata$Cell_class

latent <- fread(list.files(curr_dir, pattern = 'latent', full.names = TRUE), header=TRUE)
clusters <- runLeiden(latent, res=0.2)
plotMain(locs, by=clusters, pt.size = 1)
mdata$CVAE_clusters <- clusters
# write.csv(mdata, 'data/merfish/hypo_ani1_metadata.csv', quote=F, row.names = F)

##### DE -----------------------------------------------------------------------

# rm(list=ls())

library(DEsingle)
library(data.table)

counts <- data.frame(fread('data/merfish/hypo_ani1_counts.csv'))
metadata <- data.frame(fread('data/merfish/hypo_ani1_metadata.csv'))
locs <- data.frame(fread('data/merfish/hypo_ani1_cellcentroids.csv'))
cell_id <- paste0('cell_', 1:nrow(counts))
rownames(counts) <- cell_id

celltype <- metadata$Cell_class
groups <- metadata$CVAE_clusters
counts <- as.matrix(t(counts))

result.lst <- vector('list', length(unique(groups)))

counts_i <- counts[,which(groups == 5)]
counts_j <- counts[,which(groups == 7)]
grouping_ij <- c(rep(5, ncol(counts_i)), rep(7, ncol(counts_j)))
tmp <- DEsingle(as.matrix(cbind(counts_i, counts_j)),
                as.factor(grouping_ij))
tmp.classified <- DEtype(tmp, 0.05)
tmp.classified <- tmp.classified[which(tmp.classified$Type == 'DEg' & tmp.classified$State == 'up'),]

for (i in 1:length(unique(groups))) {
    
    message('Clusters ', i)
    grouping <- ifelse(groups == i, 1, 2)
    grouping <- as.factor(grouping)
    
    message('Run DEsingle')
    results <- DEsingle(counts, grouping)
    results.classified <- DEtype(results = results, threshold = 0.05)
    
    locsDF <- cbind(locs, clusters=grouping)
    plot <- ggplot(locsDF, aes(x = Centroid_X, 
                               y = Centroid_Y, 
                               color = clusters)) + 
        geom_point(size = 1, alpha = 0.6) +
        scale_color_manual(values = colpal(i)) +
        theme_blank()
    
    result.lst[[i]] <- list(DE=results.classified, plot=plot)
    
}

saveRDS(result.lst, 'analysis/de_results.rds')

de.lst <- result.lst[[2]]$DE
de.lst[de.lst$Type == 'DEg' & de.lst$pvalue.adj.FDR < 0.05 & de.lst$norm_foldChange > 1,]
de.plot <- result.lst[[2]]$plot; de.plot

curr.gene <- 'Gabra1'

plotGene <- function(curr.gene) plotMain(locs, by = log2(1+counts[curr.gene,]), pt.size = 1, 
         type = 'continuous', col.midpoint = 1)
plotMain(locs, by = groups)

library(Seurat)

seu <- CreateSeuratObject(counts = counts)
seu <- ScaleData(seu)
seu[['Clusters']] <- groups
seu[['Celltypes']] <- celltype
Idents(seu) <- groups

FindMarkers(seu, ident.1 = 5, ident.2 = 7)

marker_Clusters <- FindAllMarkers(seu, only.pos = TRUE)
marker_Clusters <- marker_Clusters[marker_Clusters$pct.1 > 0.6, ]
Idents(seu) <- celltype
marker_Celltype <- FindAllMarkers(seu, min.diff.pct = 0.1, only.pos = TRUE)

features <- marker_Clusters$gene
# features <- setdiff(marker_Clusters$gene, marker_Celltype$gene)

marker_Celltype
marker_Clusters
features

seu <- FindVariableFeatures(seu)
seu <- RunPCA(seu)
seu@reductions$pca@cell.embeddings[,1:2] <- as.matrix(locs[,1:2])

pdf('analysis/plots/markers.pdf', height = 40, width = 25)
FeaturePlot(seu, features = features)
dev.off()

## Hidden layer weights --------------------------------------------------------

# rm(list=ls()); graphics.off()
library(plot.matrix)
library(matrixStats)
library(data.table)
library(pals)
library(ggrepel)

counts <- data.frame(fread('data/merfish/hypo_ani1_counts.csv'))
locs <- data.frame(fread('data/merfish/hypo_ani1_cellcentroids.csv'))
encoder_weights <- fread('checkpoints/CVAE_20211107-200530/layer1_weights.csv', header=TRUE)
decoder_weights <- fread('checkpoints/CVAE_20211107-200530/layer7_weights.csv', header=TRUE)
colnames(encoder_weights) <- colnames(decoder_weights) <- colnames(counts)

pdf('analysis/plots/weights.pdf', height = 6, width = 6)
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(as.matrix(encoder_weights), border = NA, col = magma, breaks = 10,
     main = 'Encoder weights', las = 2, axis.row = list(cex = 0.5),
     xlab = '', ylab = '', cex.row = 0.4)

plot(as.matrix(decoder_weights), border = NA, col = magma, breaks = 10,
     main = 'Decoder weights', las = 2, xlab = '', ylab = '')
dev.off()

getTopFeatures <- function(w, func = 'mean') {
    genes <- colnames(w)
    w <- as.matrix(w)
    colnames(w) <- genes
    if (func == 'mean') {
        x <- colMeans(w)
    }
    if (func == 'max') {
        x <- colMaxs(w)
    }
    xdf <- data.frame(Feature=genes, Weight=x)
    xdf <- xdf[order(xdf$Weight, decreasing=TRUE),]
    rownames(xdf) <- NULL
    return(xdf)
}

ew <- getTopFeatures(encoder_weights); names(ew)[2] <- 'Encoder_weights'
dw <- getTopFeatures(decoder_weights); names(dw)[2] <- 'Decoder_weights'
mcount <- getTopFeatures(counts); names(mcount)[2] <- 'Mean_counts'
w <- Reduce(function(x,y) merge(x,y,by='Feature'), list(ew, dw, mcount))
w$Mean_weights <- rowMeans(w[,2:3])
w <- w[order(w$Encoder_weights, decreasing=F),]
w <- w[!grepl('Blank', w$Feature),]
wDF <- w
wDF$Feature[11:nrow(wDF)] <- ''
pdf('analysis/plots/ed_weights.pdf', height = 6, width = 7)
ggplot(wDF, aes(x=Encoder_weights, y=Decoder_weights, label=Feature)) + 
    geom_point(alpha = 0.8, size = 1) + 
    geom_text_repel(aes(label=Feature)) +
    theme_blank() + xlab('Encoder weights') + ylab('Decoder weights') + 
    geom_abline(slope = 1, intercept = 0, linetype = 3, col = 2)
dev.off()

## Choose a gene
gene <- 'Nts'
val <- counts[,gene]
lim <- quantile(val, 0.99)
val[val > lim] <- lim
locDF <- cbind(locs, Feature=val)
nts <- ggplot(locDF, aes(x=Centroid_X, y=Centroid_Y, col=Feature)) + 
    geom_point(alpha = 0.8) + theme_blank() +
    scale_color_gradient2(low ='blue', high ='red')

pdf('analysis/plots/sv_genes.pdf', height = 10, width = 12)
grid.arrange(mbp, chat, ucn, nts, nrow = 2, ncol = 2)
dev.off()
