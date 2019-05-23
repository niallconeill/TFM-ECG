calculate_section_parameters <- function(section){
  tci <- tci(section)
  mav <- mav(section)
  counts <- calculate_counts(section)
  exp_value <- calculate_exp(section)
  vfleak <- vfleak(section)
  section_params <-  c(tci = tci, mav = mav, count1 = counts[1], count2 = counts[2],
                       count3 = counts[3], exp=exp_value, vfleak = vfleak, label = section[1,]$label)
  
  
  
  return(section_params)
}