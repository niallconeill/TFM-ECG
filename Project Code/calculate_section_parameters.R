calculate_section_parameters <- function(section){
  tci <- tci(section)
  mav <- mav(section)
  counts <- calculate_counts(section)
  exp_value <- calculate_exp(section)
  vfleak <- vfleak(section)
  cm_jekova <- CM_Jekova(section)
  #exp_mod <- expmod(section)
  #li = Li(section)
  section_params <-  list(tci = tci, mav = mav, count1 = counts$c1, count2 = counts$c2,
                       count3 = counts$c3, exp=exp_value, vfleak = vfleak,
                       cm = cm_jekova$cm, cvbin = cm_jekova$cvbin, frqbin = cm_jekova$frqbin,
                       abin = cm_jekova$abin, kurt = cm_jekova$kurt, label = section[1,]$label)
  
  
  
  return(section_params)
}