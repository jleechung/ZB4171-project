install.packages("fitdistrplus")
library("fitdistrplus")
counts=read.csv("hypo_ani1_counts.csv")
c<-data.matrix(counts)
r<- round(c)  # round off the counts to nearest integer value 
set.seed(4171)
descdist(c[5424,],boot=1000,discrete = TRUE)     #Ependymal
par(mfrow = c(2, 2))
r5424<-fidist(r[5424,],"nbinom")
plot.legend<-c("nbinom")
denscomp(r5424, legendtext = plot.legend)
qqcomp(r5424, legendtext = plot.legend)
cdfcomp(r5424, legendtext = plot.legend)
ppcomp(r5424, legendtext = plot.legend)

c1<-c(203,433,439,472,505,516,526,555,610,696,699,711,717,759,762,778,1070,1263,1605,1610,1612,1618,1620,1622,1624,1639,1654,1661,1682,1819,2647,2708,2756,2757,3026,3352,3411,3844,3909,4144,4669,4692,5054,5091,5093,5095,5101,5125,5217,5218,5232,5508) # indexes of cell that belong to a particular cell type 
OD_mature4<-r[c1,] ## to extract rows that belong to a particular cell subtype in this case ODmature4
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

  
