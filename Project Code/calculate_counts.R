calculate_counts <- function(section){
  signal = section$signal
  
  B = c(0.5,0,-0.5)
  A = c(8, -14, 7)
  
  fSignal = filtfilt(B,A,signal)
  fSignal_abs = abs(fSignal)
  
  count1 = numeric(4)
  count2 = numeric(4)
  count3 = numeric(4)
  
  for (i in 0:3){
    intervaled_signal = fSignal[((i*samp_rate)+1):((i*samp_rate)+samp_rate)]
    max_signal = max(intervaled_signal)
    mean_signal = mean(intervaled_signal)
    meandev_signal = mean(abs(intervaled_signal - mean_signal))
    
    count1[i+1] = sum(intervaled_signal >= 0.5*max_signal)
    count2[i+1] = sum(intervaled_signal >= mean_signal)
    count3[i+1] = sum((intervaled_signal >= (mean_signal - meandev_signal)) * (intervaled_signal <= (mean_signal + meandev_signal)))
  }
  
  count1 = sum(count1)
  count2 = sum(count2)
  count3 = sum(count3)
  
  c1 = count1/4
  c2 = count2/4
  c3 = count3/4
  
  c3b = c1*c2/c3;
  return(list(c1 = c1,c2 = c2,c3 = c3b ))
}
