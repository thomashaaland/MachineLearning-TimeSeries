library(forecast)
library(knitr)

df <- read.csv("../data/INDIAVIX.csv", header=TRUE)
df <- na.omit(df)

df

close.ts <- ts(df$Close, start=c(2009, 3, 3), end=c(2020, 1, 1), frequency=256)
time <- time(close.ts)
time
n.valid <- 100
n.train <- length(close.ts) - n.valid
close.train.ts <- window(close.ts, start=time[1], end=time[n.train])
close.valid.ts <- window(close.ts, start=time[n.train + 1], end=time[n.train + n.valid])

naive.close <- naive(close.train.ts, h=20)
naive.close$mean

snaive.close <- snaive(close.train.ts, h=2)
snaive.close$mean

indexs <- c(2, 3, 5)
acc.naive.close <- accuracy(naive.close, close.valid.ts)
acc.naive.close <- acc.naive.close[, indexs]
kable(acc.naive.close)

acc.seasonal.naive.close <- accuracy(snaive.close, close.valid.ts)
acc.seasonal.naive.close <- acc.seasonal.naive.close[, indexs]
kable(acc.seasonal.naive.close)

par(mfrow=c(1,1))
plot(close.ts, main="Forecast for INDIAVIX", xlab = "Time", ylab = "Value")
lines(naive.close$mean, col=4)
lines(snaive.close$mean, col=2)
legend("topright", lty=1, col=c(4,2), legend=c("Naive method", "Seasonal naive method"))

#par(mfrow=c(2,1))
#par(pch=22, col = "black")
#plot(df_INDIAVIX[, c("Date", "Close")], type="l", main="Close", xlab="Date", ylab="Value")
#acf(df_INDIAVIX[, c("Close")], lag = 1000, main="Autocorrelation for Close")

