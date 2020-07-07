# -*R*-
# Following tutorial from Analytics Vidhya

# Load data
data("AirPassengers")
AirPassengers

# Check the datatype
class(AirPassengers)

# For ts (timeseries) start() function returns the start time
start(AirPassengers)

# For ts end() function returns the end time
end(AirPassengers)

# the frequency function returns the number of iterations for this ts over the course of a year
frequency(AirPassengers)

# Summary returns the quartiles for the time series. Good to start off with
summary(AirPassengers)

# Simple plot of the time series
par(mfrow=c(1,1))
plot(AirPassengers)

# abline adds a line to an existing plot. 
abline(reg=lm(AirPassengers ~ time(AirPassengers)))

# Cycle function displays the cycle, so 12 months for one year in this case
cycle(AirPassengers)

# Aggregates wrt the cycle with the function named in FUN
plot(aggregate(AirPassengers, FUN=mean))
lines(aggregate(AirPassengers, FUN=min))
lines(aggregate(AirPassengers, FUN=max))

# Boxplot displays the mean, quartiles and outliers With the cycle function, the data is broken down
# into the monthly categories
boxplot(AirPassengers~cycle(AirPassengers))

# Generate some random data
# use random walk we create a random walk series
y <- c(0)
for (i in seq_along(1:(2*12*30))) {
  y[i + 1] <- y[i] + (2*(sample(1:2, 1)) - 3)
}
plot(y, type="l")

par(mfrow=c(1,2))
acf(AirPassengers)
title("ACF")
pacf(AirPassengers)
title("PACF")
# Looking at the ACF plot we see that this is an AR dependency
# Looking at the PACF plot we see that there is a negative dependency of x[i+1] and x[i + 11], and a positive dependency of x[i + 8]
# So, function should be something like x[i] = a*x[i-1] + b*x[i-2] + c*x[i-9] + d*x[i-13]
# using the three significant contributions, we have by inspection

# Generate some random data
# use random walk we create a random walk series
y <- c(0)
a = 0.9
b = -0.12
c = 0.125
d = -0.5
for (i in seq_along(1:200)) {
  move <- (2*(sample(1:2, 1)) - 3)
  y[i + 1] <- a*y[i] + move
  if (i > 1) {
    y[i + 1] <- y[i+1] + b*y[i-1]
  }
  if (i > 8) {
    y[i+1] <- y[i+1] + c*y[i-8]
  }
  if (i > 12) {
    y[i+1] <- y[i+1] + d*y[i-12]
  }
}
par(mfrow=c(1,1))
plot(y, type="l")
title("Sample seasonality plot with random walk")

# Framework to generally apply:
# 1: Visualize timeseries
# 2: Make the time series stationary
# 3: plot ACF  (AutoCorrolationFunction) / PACF (PartialAutoCorrelationFunction)
# 4: Build ARIMA model (AutoRegressive Integration MovingAverage)
# 5: Make the predictions

#------------------------------
# 1: Visualize
par(mfrow=c(1,2))
plot(AirPassengers)
title("Passengers")
plot(log(AirPassengers))
title("Passengers log")
# Notes: From the visualization there is a trend, there is a cycle about twice every year
# The series does not have heterodacity (VAR is a function of time)

#-----------------------------
# 2: Make the timeseries stationary:
# First, the adf test (Augmented Dickey Fuller)
tseries::adf.test(diff(log(AirPassengers)), alternative="stationary", k=0)
# Appears to be stationary 'enough'

#-----------------------------
# 3: plot ACF and PACF
par(mfrow=c(1,2))
acf(log(AirPassengers), main="AutoCorrelation")
pacf(log(AirPassengers), main="Partial AutoCorrelation")
# Noticing that ACF does not decrease quickly, so not stationary 'enough'
# Need to regress to the mean
# ACF and PACF on diff
par(mfrow=c(1,2))
acf(diff(log(AirPassengers)), main="ACF on difference on log")
pacf(diff(log(AirPassengers)), main="PACF on difference on log")
# Since ACF cuts off after first iteration, p=0.

#------------------------------
# 4: Build Arima
fit <- arima(log(AirPassengers), c(0,1,1), seasonal = list(order = c(0, 1, 1), period = 12))
pred <- predict(fit, n.ahead = 10*12)
par(mfrow=c(1,1))
ts.plot(AirPassengers, exp(1)^pred$pred, log = "y", lty=c(1,3), main="Prediction 10 years ahead")
