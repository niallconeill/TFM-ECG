calculate_exp <- function(section){
  signal = section$signal 
  L  = length(signal)
  max_val = max(abs(signal))
  
  max_index = which.max(abs(signal))
  n =  1:L
  tau = 3
  
  es = max_val*exp(-(abs(max_index-n)/(tau*samp_rate)))
  
  above = es >= signal

  inter = which(diff(above) != 0)
  
  N = length(inter)*60/8
  
  return(N)
}