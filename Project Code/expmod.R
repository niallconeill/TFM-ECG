expmod <- function(section){
  signal <- section$signal
  
  #Normalize
  max = max(signal)
  signal = signal/max
  
  #Peak Detection
  nm = peakdet(signal,0.2)
  mxpos = nm$maxtab[,1]
  
  if(mxpos[1] == 1){
    mxpos = mxpos[2:length(mxpos)]
  }
  
  tau = 0.2
  L = length(signal)
  
  if(is.null(mxpos)){
    N = 1*60/(L/samp_rate)
    return(N)
  }
  
  #Initialization of first maxima
  nm = mxpos
  nm1 = nm[1]
  En = numeric(L)
  En[1:nm1-1] = signal[1:nm1-1]
  maxpos = nm1
  
  # Remaining samples
  n = seq(nm1,L,1)
  nmj = nm1
  N = 1
  fin = 0
  
  while(fin == 0){
    Mj = signal[nmj]
    En[nm1:L] = Mj * exp(-(n-nmj)/(tau*samp_rate))
    
    ncj = which(signal > En)
    val = which(ncj-nmj > 10)
    ncj = ncj[val]
    
    if(is.null(ncj) |  length(ncj) == 0){
      fin = 1
    }
    else{
      ncj = ncj[1]
      aux = signal
      aux[1:ncj-1] = numeric(ncj-1)
      nm = peakdet(aux,0.3)
      
      if(is.null(nm) | length(nm) == 0){
        fin = 1
        En[ncj:L] = signal[ncj:L]
      }
      else{
        nmj = nm$maxtab[nm$maxtab>ncj,1]
        
        if(is.null(nmj) | length(nmj) == 0){
          fin = 1
          En[ncj:L] = signal[ncj:L]
        }
        else{
          nmj = nm$maxtab[nm$maxtab>ncj,1]
          nmj = nmj[1]
          En[ncj:nmj-1] = signal[ncj:nmj-1]
          n = nmj:L
          
          
          N = N + 1
          
          maxpos = merge(maxpos,nmj)
          #maxpos = as.vector(rbind(maxpos,nmj))
        }
      }
    }
  }
  
  N = N*60/(L/samp_rate)
  
  return(N)
}