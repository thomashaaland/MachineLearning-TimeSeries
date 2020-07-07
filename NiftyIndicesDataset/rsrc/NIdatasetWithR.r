#-*R*-
# Forecasting for the NiftyIndicesDataset
# First, load the dataset
NiftyIndices.df <- read.csv("../data/INDIAVIX.csv", header = TRUE, stringsAsFactors = FALSE)

# Create a zoo object
NiftyIndices.z <- zoo(x = NiftyIndices.df$Close, order.by = as.Date(NiftyIndices.df$Date))

# Framework to generally apply:
# 1: Visualize timeseries
par(mfrow = c(1,2), bg = "white")
plot(NiftyIndices.z, main="NiftyIndices value")
plot(log(NiftyIndices.z), main="Log")
# Appears to be a cycle once less than each year

# 2: Make the time series stationary
# Augmented Dickey Fuller
tseries::adf.test(log(NiftyIndices.z), alternative = "stationary", k=0)
# Appears stationary, since p < 0.01

# 3: plot ACF  (AutoCorrolationFunction) / PACF (PartialAutoCorrelationFunction)
# Check Autocorrolation first
par(mfrow = c(1,2))
acf(log(NiftyIndices.z), na.action = na.pass, main = "AC")
pacf(log(NiftyIndices.z), na.action = na.pass, main = "PAC")

# Since it is nor stationary, lets try diff
par(mfrow = c(1,2))
acf(diff(log(NiftyIndices.z)), na.action = na.pass, main = "diff AC")
pacf(diff(log(NiftyIndices.z)), na.action = na.pass, main = "dff PAC")

P <- 2
I <- 0
Q <- 0

# 4: Build ARIMA model (AutoRegressive Integration MovingAverage)
fit <- arima(log(NiftyIndices.z), c(P, I, Q), seasonal = c(P, I, Q))

# 5: Make the predictions
ahead = 100
pred <- predict(fit, n.ahead = 3)
DF <- as.data.frame(pred)

endDate <- index(tail(NiftyIndices.z, 1))

seq(from=endDate + 1, to=endDate+ahead, "days")

pred <- zoo(DF, seq(from=endDate + 1, to=endDate+ahead, "days"))$pred

par(mfrow=c(1,1))
plot(pred)

#ts.plot(NiftyIndices.z, exp(1)^pred$pred, log = "y", lty=c(1,3), main="Prediction 3 days ahead")
