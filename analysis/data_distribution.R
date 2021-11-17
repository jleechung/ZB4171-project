install.packages("fitdistrplus")
library("fitdistrplus")
counts=read.csv("hypo_ani1_counts.csv")
c<-data.matrix(counts)
r<- round(c)  # round off the counts to nearest integer value 
set.seed(4171)
descdist(r[5424,],boot=1000,discrete = TRUE)     #Ependymal, the value can be changed to plot cells from other celltypes
par(mfrow = c(2, 2))
r5424<-fidist(r[5424,],"nbinom")
plot.legend<-c("nbinom")
denscomp(r5424, legendtext = plot.legend)
qqcomp(r5424, legendtext = plot.legend)
cdfcomp(r5424, legendtext = plot.legend)
ppcomp(r5424, legendtext = plot.legend)

R=read.csv("hypo_ani1_counts1.csv")            #csv with cell_class column from metadata added.
ODMature4<-subset(R,Cell_class=="OD Mature 4") 
ODMature4<-subset(ODMature4,select=-c(Cell_class))
ODMature1<-subset(R,Cell_class=="OD Mature 1") 
ODMature1<-subset(ODMature1,select=-c(Cell_class))
ODMature2<-subset(R,Cell_class=="OD Mature 2") 
ODMature2<-subset(ODMature2,select=-c(Cell_class))
Endo3<-subset(R,Cell_class=="Endothelial 3")
Endo3<-subset(Endo3,select=-c(Cell_class))
Endo2<-subset(R,Cell_class=="Endothelial 2")
Endo2<-subset(Endo2,select=-c(Cell_class))
Endo1<-subset(R,Cell_class=="Endothelial 1")
Endo1<-subset(Endo1,select=-c(Cell_class))
Excitatory<-subset(R,Cell_class=="Excitatory")
Excitatory<-subset(Excitatory,select=-c(Cell_class))
Microglia<-subset(R,Cell_class=="Microglia")
Microglia<-subset(Microglia,select=-c(Cell_class))
Inhibitory<-subset(R,Cell_class=="Inhibitory")
Inhibitory<-subset(Inhibitory,select=-c(Cell_class))
Ambiguous<-subset(R,Cell_class=="Ambiguous")
Ambiguous<-subset(Ambiguous,select=-c(Cell_class))
Peri<-subset(R,Cell_class=="Pericytes")
Peri<-subset(Peri,select=-c(Cell_class))
Astrocyte<-subset(R,Cell_class=="Astrocyte")
Astrocyte<-subset(Astrocyte,select=-c(Cell_class))



install.packages(matrixStats)        
library(matrixStats)
Mean<-colMeans(r)
Variance<-(colSds(r))^2
plot(log2(Mean+1), log2(Variance+1),pch=19, cex=0.5)
abline(a=0,b=1,lty=3,col=2)
 

ll_nbinom<-rep(NA, nrow(r))
ll_pois<-rep(NA, nrow(r))
for (i in 1:nrow(r)) {
  if (i %% 100 == 0) message('Iteration ', i)
  fit_nbinom <- fitdist(r[i,],'nbinom')
  fit_pois <- fitdist(r[i,],'pois')
  ll_nbinom[i] <- summary(fit_nbinom)[[6]]
  ll_pois[i] <- summary(fit_pois)[[6]]
}
plot(y=ll_pois, x=ll_nbinom,pch=19,cex=0.5,col=rgb(0,0,0,0.5))
abline(a=0,b=1,col=2)


BIC_nbinom<-rep(NA, nrow(r))                                                  
BIC_pois<-rep(NA, nrow(r))
for (i in 1:nrow(r)) {
  if (i %% 100 == 0) message('Iteration ', i)
  fit_nbinom <- fitdist(r[i,],'nbinom')
  fit_pois <- fitdist(r[i,],'pois')
  BIC_nbinom[i] <- summary(fit_nbinom)[[8]]
  BIC_pois[i] <- summary(fit_pois)[[8]]
}
plot(y=BIC_pois, x=BIC_nbinom,pch=19,cex=0.5,col=rgb(0,0,0,0.5))
abline(a=0,b=1,col=2)
