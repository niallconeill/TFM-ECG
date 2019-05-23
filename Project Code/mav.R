mav <- function(section){
  
  signal = section$signal
  
  mav = mean(abs(signal))
  
  return(mav)
}


