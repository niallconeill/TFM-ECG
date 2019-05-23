vfleak <- function(section){
  signal = section$signal
  end = length(signal)
  
  sumv = sum(abs(signal[2:end]))
  sumvdiff = sum(abs(signal[2:end] - signal[1:(end-1)]))
  
  N = floor((pi * (sumv)/(sumvdiff)) + 1/2)
  
  num = sum(abs(signal[(N+1):end] + signal[1:(end-N)]))
  den = sum(abs(signal[(N+1):end]) + abs(signal[1:(end-N)]))
  
  vf = num/den
  
  return(vf)
}