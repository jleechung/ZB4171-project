install.packages("fitdistrplus")
library("fitdistrplus")
counts=read.csv("hypo_ani1_counts.csv")
c<-data.matrix(counts)
r<- round(c)  # round off the counts to nearest integer value 
set.seed(4171)
descdist(c[5424,],boot=1000,discrete = TRUE)     #Ependymal
fidist(r[5424,],"nbinom")
