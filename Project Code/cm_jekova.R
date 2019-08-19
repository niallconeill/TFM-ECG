CM_Jekova <- function(section){
  wL = 4
  signal = section$signal
  
  n = length(signal)
  s = numeric(n)
  
  xi = signal - mean(signal)
  
  xmax = max(xi)
  xmin = min(xi)
  
  Pc = length(which(0 < xi & xi < 0.1*xmax))
  Nc = length(which(0.1*xmin < xi & xi < 0))
  
  if(Pc + Nc < 0.4*n){
    th = 0 
  }
  else if(Pc < Nc){
    th = 0.2*xmax
  }
  else{
    th = 0.2*xmin
  }
  
  s[xi >= th] = 1
  
  # Complexity 
  k = kolmogorov(s)
  cm = k[2]
  
  # Covariance 
  cvbin = var(s)
  
  #Frequency 
  frqbin = sum(diff(s, differences = 1) == 1)
  frqbin = frqbin / wL
  
  #Area
  N = sum(s)
  abin = max(N,samp_rate*wL-N)
  
  #Kurtosis
  m_s = mean(signal)
  s_s = sd(signal)
  kurt = mean((signal-m_s)^4)/s_s^4 - 3
  
  return(list(cm = cm ,cvbin = cvbin ,frqbin = frqbin ,abin = abin, kurt = kurt))
}