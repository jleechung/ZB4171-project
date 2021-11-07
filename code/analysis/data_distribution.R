install.packages("fitdistrplus")
library("fitdistrplus")
counts=read.csv("hypo_ani1_counts.csv")
c<-data.matrix(counts)
r<- round(c)  # round off the counts to nearest integer value 
set.seed(4171)
descdist(c[1,],boot=1000,discrete = TRUE)
fidist(r[1,],"nbinom")
