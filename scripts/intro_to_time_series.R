# Introduction to time series
# Neil Rankin (neil@predictiveinsights.net)
# 2019/07/25

# load packages

library(prophet)
library(tidyverse)
library(lubridate)
library(quantmod)

# store sales come from the TSA package (for time series analysis)
library(TSA)
data("retail")

# can look at some of the aspects of this data
retail
frequency(retail)

start(retail)

end(retail)

plot(retail)

monthplot(retail)

time_1 <- time(retail)

# zoo is another time/date package
library(zoo)
time_2 <- as.yearmon(time(retail))


ts <- as.data.frame(time_2) %>% 
  rename(ds = time_2) 

ts_df <- bind_cols(ts, as.data.frame(retail)) %>% 
  rename(y = Sales)

# look at the data
ggplot(ts_df, aes(y = y, x = ds)) + 
  geom_line()


# Now we can use prophet

m <- prophet(ts_df, daily.seasonality = TRUE)
summary(m)


# make a future data frame
future <- make_future_dataframe(m , periods = 365)

# and predict
pred <- predict(m, future)

# look at each component

ggplot(pred, aes(y = yhat, x = ds)) + 
  geom_line(group = 1) + 
  geom_line(aes(y = trend)) + 
  geom_line(aes(y = additive_terms))

# Let's now follow an ARIMA approach
# Another reference:
# https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials

library(forecast)
adf.test(ts_df$y)

Acf(ts_df$y, main='')
Pacf(ts_df$y, main='')

count_d1 = diff(ts_df$y, differences = 1)
plot(count_d1)
adf.test(count_d1)


Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')


# from forecast
fit <- auto.arima(ts_df$y, seasonal=FALSE)
fit

# Have a look at the residuals
tsdisplay(residuals(fit), lag.max=45, main='(2,1,2) Model Residuals')

dy <- ts_df %>% 
  mutate(dy = y - lag(y), 
         dy12 = y - lag(y, 12)) %>% 
  filter(!is.na(dy12))

fit1 <- auto.arima(dy$dy1, seasonal=FALSE, max.p = 24, max.q = 24)
fit1
tsdisplay(residuals(fit1), lag.max=45, main='fit1 Model Residuals')

fit12 <- auto.arima(dy$dy12, seasonal=FALSE, max.p = 24, max.q = 24)
fit12
tsdisplay(residuals(fit12), lag.max=45, main='fit12 Model Residuals')

#fit<-auto.arima(ts_df$y, seasonal=FALSE)
tsdisplay(residuals(fit12), lag.max=45, main='(1,0,4) Model Residuals')

# Exercise
# Can you build an ARIMA model for stock prices?

quantmod::getSymbols("AMZN",from="2008-08-01",to="2018-08-17")
# AMZN_log_returns<-  AMZN %>% Ad() %>% dailyReturn(type='log')

plot(AMZN$AMZN.Adjusted)

fit_a <- auto.arima(AMZN$AMZN.Adjusted, seasonal=FALSE)
fit_a

# Try and build your model for a 'training' period and then forecast out-of-sample
# How well does it do?

