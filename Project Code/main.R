################# Preprocessing of Data Files ###############
#############################################################


################ Set up Environment #########################
setwd("~/Big Data Masters/Trabajo Fin De Masters/Preprocessing Code")
wrk_dir <- "C:/Users/niall/OneDrive/Documents/Big Data Masters/Trabajo Fin De Masters/"

#First load the signal library 
library("signal")
library("zoo")

############## Source #######################################

source("preprocess_signal.R")
source("generate_params_df.R")
source("calculate_section_parameters.R")
source("tci.R")
source("mav.R")
source("calculate_counts.R")
source("calculate_exp.R")
source("vfleak.R")


#############################################################

#Set array of folders names to loop through
folders <- c("CUDB Data/","VFDB Data/")
sampling_rate <- c(250,360)

#Shockable Labels 
shockable <- c("(VF", "[", "]", "(VT", "(VFL", "(VFIB")

for (i in 1:length(folders)){
  files <- list.files(path = paste(wrk_dir,folders[i],'Data/', sep =""), pattern ="*.csv", full.names=FALSE, recursive = FALSE)
  samp_rate <- sampling_rate[i]
  for (file in files){
    preproc_data <- preprocess_signal(file)
    processed_file <- generate_params_df(preproc_data)
    assign(substr(file,1,  nchar(file)-4), processed_file)
    
  }
}






