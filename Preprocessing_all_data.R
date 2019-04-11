################# Preprocessing of Data Files ###############
#############################################################

wrk_dir <- "C:/Users/niall/OneDrive/Documents/Big Data Masters/Trabajo Fin De Masters/"

#First load the signal library 
library("signal")


#Set array of folders names to loop through
folders <- c("CUDB Physical/","VFDB Physical/")
sampling_rate <- c(250,360)

for (i in 1:length(folders)){
  print(i)
}

for (i in 1:length(folders)){
  files <- list.files(path = paste(wrk_dir,folders[i], sep =""), pattern ="*.csv", full.names=FALSE, recursive = FALSE)
  for (file in files){
    
    #Load CSV file 
    temp <- read.csv(paste(wrk_dir,folders[i],file, sep =""), sep = ";", header = FALSE)
    temp[] <- lapply(temp, function(x) as.numeric(as.character(x)))
    
    #Remove NAs from file
    temp <- na.omit(temp)
    
    #Mean subtraction 
    temp[,2] <- temp[,2] - mean(temp[,2])
    
    #Create a moving average filter of order 5
    mA_filt <- Ma(5)
    
    #Apply the filter to the data
    data_Ma <- filter(mA_filt, temp[,2])
    
    #Generate a high pass butterworth filter 
    fc <- 1 #Cut-off frequency 
    f_samp <- sampling_rate[i] # Sampling frequency 
    
    W <- fc/(f_samp/2) #Formula from Matlab documentation example 
    
    high_filt <- butter(5,W, type = "high")
    
    #Apply high-pass filter to data
    data_high <- filter(high_filt,data_Ma)
    
    #Create a low pass buttorworth filter
    fc <- 30
    
    W <- fc/(f_samp/2)
    
    low_filt <- butter(5, W, type = "low")
    
    #Apply low-pass filter to data
    data_low <- filter(low_filt,data_high)
    
    temp[,2] <- data_low
    
    assign(file, temp[,0:2])
  }
}











