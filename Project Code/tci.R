tci <- function(section){
  
  tci6 = numeric(6)
  
    
    for (j in 1:6){
      becg1 = numeric(samp_rate)
      becg2 = numeric(samp_rate)
      becg3 = numeric(samp_rate)
      
      
      start1 <- ((j-1) * samp_rate)+1 
      end1 <- j * samp_rate
      second1 <- section[start1:end1,]
      
      start2 <- (j * samp_rate)+1 
      end2 <- (j+1) * samp_rate
      second2 <- section[start2:end2,]
      
      start3 <- ((j+1) * samp_rate)+1 
      end3 <- (j+2) * samp_rate
      second3 <- section[start3:end3,]
      
      stage1 = second1$signal - mean(second1$signal)
      maxv = max(stage1)
      th1 = 0.2*maxv 
      becg1[stage1>th1] = 1 
      
      stage2 = second2$signal - mean(second2$signal)
      maxv = max(stage2)
      th2 = 0.2*maxv 
      becg2[stage2>th2] = 1 
      
      stage3 = second3$signal - mean(second3$signal)
      maxv = max(stage3)
      th3 = 0.2*maxv 
      becg3[stage3>th3] = 1 
      
      becg = c(becg1, becg2, becg3)
      
      aux = c(0, diff(becg, differences = 1))
      
      s1 = which(aux[1:samp_rate]== -1)
      
      if (length(s1) == 0){
        t1 = 1
      }
      else{
        t1 = (samp_rate-s1[length(s1)])/samp_rate
      }
      
      s2 = aux[(samp_rate+1):(2*samp_rate)]
      index = which(s2 != 0)
      
      pulses = s2[index]
      
      if(pulses[1] == -1 && pulses[length(pulses)] == -1){
        t2 = 0 
        t3 = (samp_rate - index[length(index)])/samp_rate
        N = (length(pulses) + 1)/2
      }
      else if(pulses[1] == 1 && pulses[length(pulses)] == 1){
        t2 = index[1]/samp_rate
        t3 = 0 
        N = (length(pulses) + 1)/2
      } 
      else if (pulses[1] == -1 && pulses[length(pulses)] == 1){
        t2 = 0 
        t3 = 0
        N = (length(pulses) + 2)/2
      }
      else if (pulses[1] == 1 && pulses[length(pulses)] == -1 ){
        t2 = index[1]/samp_rate
        t3 = (samp_rate - index[length(index)])/samp_rate
        N = (length(pulses))/2
      }
      else{
        print("This should not be happening")
      }
      
      s4 = which(aux[(2*samp_rate+1):(3*samp_rate)] == 1)
      if (length(s4) == 0){
        t4 = 1
      }
      else{
        t4 = s4[1]/samp_rate
      }
      
      tci6[j] = 1000/((N-1)+(t2/(t1+t2)) + (t3/(t3+t4)))
    }
  
  tci = mean(tci6)
  
  return(tci)
}