library(monitoR)
library(warbleR)
library(Rraven)

source('CrossCorrelationResample.R')

#########################
#Birds
#The pattern file needs to be in the same folder as the audio recordings
#
recdir<-"G:/Piha/"
recdir<-"G:/CARPETA-GRABADORES MADERACRE-OTORONGO-CHULLACHAQUI-2019/GRABADOR-SDZG-AM-5"
patfile<-"Temp_Lipaugus-vociferans-520937.wav"

#number of cores
ncore=6

#load recordings in directory
seltab<-cbind(selection_table(path=recdir,whole.recs = T),info_wavs(recdir)[,-1])

#matrix to match template against all recordings
cm <- as.matrix(data.frame(pattern=patfile, file=paste(seltab$sound.files[-which(seltab$sound.files==patfile)],seltab$selec[-which(seltab$sound.files==patfile)],sep="-"),stringsAsFactors = F))

#cross-correlation for all recordings
#bp is a band-pass filter to only use a certain frequency range
xc.output <- cross_correlation2(X = data.frame(seltab), output = "list",bp=c(0,8),ovlp=50, 
                   compare.matrix = cm, path = recdir,parallel = ncore,step = 4)
hist(xc.output$scores$score)

#find peaks
xcpks <- find_peaks(xc.output = xc.output, path = recdir,parallel = ncore,cutoff=0.3)

xcpks<-overlapping_sels(xcpks,drop=T,priority.col = "score",max.ovlp = 1,parallel = ncore)



#writ spectrograms with regions selected
spectrograms(X = xcpks, wl = 1024, by.song = "sound.files", flim=c(0,8),ovlp=50, fill=adjustcolor("#d90016", alpha.f = 0.15),
             collevels = seq(-80, 0, 5), xl = 3, pb = T, path = recdir,parallel = ncore)

#export for raven
exp_raven(xcpks,path=recdir,file.name = "phia_all.txt",sound.file.path=recdir)

#export Kaleidoscope
write.kaleidoscope(xcpks,path=recdir,file="annotations2.csv")





#########################
#Bats
#The pattern file needs to be in the same folder as the audio recordings
#
recdir<-"G:/AM-26 Bats"
patfile<-"Bat1b_20190619_014000.WAV"

recdir<-"G:/CARPETA-GRABADORES MADERACRE-OTORONGO-CHULLACHAQUI-2019/GRABADOR-SDZG-AM-9"
patfile<-"Bat_AM9_20190627_134000.wav"

#number of cores
ncore=6 

#load recordings in directory
seltab<-cbind(selection_table(path=recdir,whole.recs = T),info_wavs(recdir)[,-1])

#matrix to match template against all recordings
cm <- as.matrix(data.frame(pattern=patfile, file=paste(seltab$sound.files[-which(seltab$sound.files==patfile)],seltab$selec[-which(seltab$sound.files==patfile)],sep="-"),stringsAsFactors = F))

#only use if you want to limit the number of recordings
cm<-cm[1:300,]
cm<-cm[2300:2800,]

#cross-correlation for all recordings
#bp is a band-pass filter to only use a certain frequency range
xc.output <- cross_correlation2(X = data.frame(seltab), output = "list",bp=c(35,50),ovlp=50, 
                                compare.matrix = cm, path = recdir,parallel = ncore,step=2)
hist(xc.output$scores$score)
max(xc.output$scores$score)

#find peaks (adjust cutoff)
xcpks <- find_peaks(xc.output = xc.output, path = recdir,parallel = 1,cutoff = 0.35)
dim(xcpks)
hist(xcpks$score)

xcpks<-overlapping_sels(xcpks,drop=T,priority.col = "score",max.ovlp = 1,parallel=1)

#write spectrograms with regions selected
spectrograms(X = xcpks, wl = 1024, by.song = "sound.files", flim=c(30,50),ovlp=50, fill=adjustcolor("#d90016", alpha.f = 0.15),
             collevels = seq(-80, 0, 5), xl = 3, pb = T, path = recdir,parallel = ncore)

#export for raven
exp_raven(xcpks,path=recdir,file.name = "raven.txt",sound.file.path=recdir)

#export Kaleidoscope
write.kaleidoscope(xcpks,path=recdir,file="annotations.csv")

cut_sels(xcpks[sample(nrow(xcpks),100),],path=recdir,dest.path = "G:/AM-26 Bats/clip")

#look at scoares over time
scores<-xc.output$scores
scores$DateTime<-as.POSIXct(gsub(".WAV-1","",scores$sound.file),format="%Y%m%d_%H%M%OS")
scores$Time<-sapply(scores$DateTime,function(x){
  x<-as.numeric(strsplit(format(x,"%H:%M:%S"),":")[[1]])
  x[1]+x[2]/60+x[3]/3600
})

scores$TimeLocal<-scores$Time-5
scores$TimeLocal[scores$TimeLocal<0]<-scores$TimeLocal[scores$TimeLocal<0]+24


#detection summary by time
scores<-data.frame(file=unique(as.character(xc.output$scores$sound.file)),mean=tapply(xc.output$scores$score,xc.output$scores$sound.files,mean),max=tapply(xc.output$scores$score,xc.output$scores$sound.files,max),count=tapply(xc.output$scores$score>0.4,xc.output$scores$sound.files,sum))
scores$DateTime<-as.POSIXct(gsub(".WAV-1","",scores$file),format="%Y%m%d_%H%M%OS")
scores$Time<-sapply(scores$DateTime,function(x){
  x<-as.numeric(strsplit(format(x,"%H:%M:%S"),":")[[1]])
  x[1]+x[2]/60+x[3]/3600
})
scores$TimeLocal<-scores$Time-5
scores$TimeLocal[scores$TimeLocal<0]<-scores$TimeLocal[scores$TimeLocal<0]+24
scores$Date<-as.Date(scores$DateTime)

plot(sort(unique(floor(scores$TimeLocal))),tapply(scores$mean,floor(scores$TimeLocal),mean),type="l",xlab="Time",ylab="Score")
plot(sort(unique(floor(scores$TimeLocal))),tapply(scores$max,floor(scores$TimeLocal),mean),type="l",xlab="Time",ylab="Score")
plot(sort(unique(floor(scores$TimeLocal))),tapply(scores$count,floor(scores$TimeLocal),mean),type="l",xlab="Time",ylab="Score")


plot(as.numeric(scores$DateTime),scores$count)

t<-lapply(sort(unique(scores$Date))[1:20],function(x)tapply(scores$count[scores$Date==x],floor(scores$Time[scores$Date==x]),mean))
t<-do.call(rbind,t)
heatmap(t, Colv = NA, Rowv = NA,labRow = sort(unique(data$Date[data$Code==code])))
