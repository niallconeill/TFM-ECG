kolmogorov <- function(signal){
  
  n = length(signal)
  c = 1
  l = 1
  
  i = 0 
  k = 1
  k_max = 1
  stop = 0 
  
  while (stop == 0){
    if(signal[i+k] != signal[l+k]){
      if(k > k_max){
        k_max = k 
      }
      i = i+1
      
      if(i == 1){
        c = c+1
        l = l+k_max
        if(l+1 > n){
          stop = 1
        }
        else{
          i = 0
          k = 1
          k_max = 1
        }
      }
      else{
        k = 1
      }
    }
    else{
      k = k+1
      if (l+k > n){
        c = c+1
        stop = 1
      }
    }
  }
  
  b = n/log2(n)
  
  complexity = c/b
  kolmogorov = c(c,complexity)
}