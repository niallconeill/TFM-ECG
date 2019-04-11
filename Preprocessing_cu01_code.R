#load signal data 

data <- read.csv("C:/Users/niall/OneDrive/Documents/Big Data Masters/Trabajo Fin De Masters/CUDB Physical/cu01_data.csv", sep = ";", header = FALSE)
plot(data)

data[,2] <- data[,2] - mean(data[,2]) #mean subraction

#preprocessing the data using the signal library 

library("signal")

#Create a moving average filter of order 5
mA_filt <- Ma(5)

#Apply the filter to the data
data_Ma <- filter(mA_filt, data[,2])
plot(data_Ma)

#Generate a high pass butterworth filter 
fc <- 1 #Cut-off frequency 
f_samp <- 250 # Sampling frequency 

W <- fc/(f_samp/2) #Formula from Matlab documentation example 

high_filt <- butter(5,W, type = "high")

#Apply high-pass filter to data
data_high <- filter(high_filt,data_Ma)

plot(data_high)

#Create a low pass buttorworth filter
fc <- 30

W <- fc/(f_samp/2)

low_filt <- butter(5, W, type = "low")

#Apply low-pass filter to data
data_low <- filter(low_filt,data_high)

plot(data_low)
