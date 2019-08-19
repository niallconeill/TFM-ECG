preprocess_signal <- function(file){
  
  #Load data CSV file 
  temp <- read.csv(paste(wrk_dir,folders[i],'Data/', file, sep =""), sep = ";", header = FALSE, stringsAsFactors = FALSE)
  temp[] <- lapply(temp, function(x) as.numeric(as.character(x)))
  temp$ID <- seq.int(nrow(temp))
  
  #Load the annotation file
  annot <- read.csv(paste(wrk_dir, folders[i], 'Annotations/',file, sep = ""), sep = ";", header = FALSE, stringsAsFactors = FALSE)
  annot <- annot[,2:3]
  colnames(annot) <- c("ID", "label")
  
  #Merge the annotations to the data temp file
  temp <- merge(temp,annot, by="ID", all =TRUE)
  y <- na.locf(temp[,4], na.rm = FALSE)
  temp[,4] <- y 
  
  #Label noise and artifacts
  temp$label[temp$label %in% c("(NOISE","~", "|", "(ASYS")] <- NA
  
  #Relabel as either shockable or non-shockable 
  temp$label[temp$label %in% shockable] <- "Sh"
  temp$label[!temp$label == "Sh" & !temp$label == "NOISE"] <- "NSh"
  
  #Remove NAs from file
  temp <- subset(temp, !is.na(temp$V2))
  
  #Mean subtraction 
  temp[,3] <- temp[,3] - mean(temp[,3], na.rm = TRUE)
  
  #Create a moving average filter of order 5
  mA_filt <- Ma(5)
  
  #Apply the filter to the data
  data_Ma <- signal::filter(mA_filt, temp[,3])
  
  #Generate a high pass butterworth filter 
  fc <- 1 #Cut-off frequency 
  f_samp <- samp_rate # Sampling frequency 
  
  W <- fc/(f_samp/2) #Formula from Matlab documentation example 
  
  high_filt <- butter(5,W, type = "high")
  
  #Apply high-pass filter to data
  data_high <- signal::filter(high_filt,data_Ma)
  
  #Create a low pass buttorworth filter
  fc <- 30
  
  W <- fc/(f_samp/2)
  
  low_filt <- butter(5, W, type = "low")
  
  #Apply low-pass filter to data
  data_low <- signal::filter(low_filt,data_high)
  
  temp[,3] <- data_low
  colnames(temp) <- c("ID", "time", "signal", "label")
  return(temp)
  
}