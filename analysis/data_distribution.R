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

R=read.csv("hypo_ani1_counts.csv1")
OD_mature4<-subset(R,Cell_class=="OD Mature 4")
rowOD_mature4<-rowSums(OD_mature4)
quantile(rowOD_mature4)
index<-numeric(20)
q1=0
q2=0
q3=0
q4=0
c=0
while(c<20){
for (i in 1:length(rowOD_mature4)){
  if(rowOD_mature4[i]<=292.5 & count1<5){
    q1=q1+1
    c=c+1
    index[c]=i
  }else if(rowOD_mature4[i]<=339.5 & rowOD_mature4[i]>292.5 & count2<5){
    q2=q2+1
    c=c+1
    index[c]=i
  }else if(rowOD_mature4[i]>339.5 & rowOD_mature4[i]<=380.0 & count3<5){
    q3=q3+1
    c=c+1
    index[c]=i
  }else if(count4<5 & rowOD_mature4[i]>380.0){
    q4=q4+1
    c=c+1
    index[c]=i
  }
}
}
r[index,] #to get indexes of the 20 sampled cells from the original count matrix.

install.packages(matrixStats)        
library(matrixStats)
Mean<-colMeans(r)
Variance<-(colSds(r))^2
plot(log2(Mean+1), log2(Variance+1),pch=19, cex=0.5)
abline(a=0,b=1,lty=3,col=2)
 
