# Randomisation class exercise
# Neil Rankin (neil@predictiveinsights.net)
# 2019/07/29

library(tidyverse)
library(googlesheets)


# LOAD DATA & WRANGLE ----
# list files - to identify file
# gs_ls()

# identify key for google sheet (the sequence after /d/ amd excluding from /edit onwards)
# https://docs.google.com/spreadsheets/d/17YZeH-mVn3kb1249isS_rh70hraMPN3QLAVCc2l-SZQ/edit#gid=0

gs <- gs_key("17YZeH-mVn3kb1249isS_rh70hraMPN3QLAVCc2l-SZQ", lookup=FALSE, visibility="private")

df <- gs_read(gs)


# look just at Maths

df_m <- df %>% 
  filter(Component == "M")

test1 <- lm(Score ~ Treatment, df_m)
summary(test1)

