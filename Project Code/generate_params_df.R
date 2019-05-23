generate_params_df <- function(preproc_data){
  section_size <- 8 * samp_rate #Calculate number of data points in one 8s window
  
  no_sections <- nrow(preproc_data)%/%section_size
  
  
  df <- data.frame(tci = rep(NA, no_sections), mav = rep(NA, no_sections), count1 = rep(NA, no_sections), count2 = rep(NA, no_sections),
                   count3 = rep(NA, no_sections), exp = rep(NA,no_sections), vfleak = rep(NA, no_sections), label = rep("",no_sections), stringsAsFactors = FALSE)
  
  for (j in 1:no_sections){
    start <- ((j-1) * section_size)+1 
    end <- j * section_size
    section <- preproc_data[start:end,]
    
    if(!anyNA(section) | !length(unique(section$label)) > 1){
      section_params <- calculate_section_parameters(section)
      
      df <- rbind(df,section_params)
      
    }
  }
  
  df <- df[complete.cases(df),]
  
  return(df)
}