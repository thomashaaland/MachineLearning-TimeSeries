df_INDIAVIX <- read.csv("../data/INDIAVIX.csv", header = TRUE);
#rownames(df_INDIAVIX) <- df_INDIAVIX$Date
#df_INDIAVIX$Date <- NULL
head(df_INDIAVIX)

par(mfrow=c(2,1))
par(pch=22, col = "black")
plot(df_INDIAVIX[, c("Date", "Close")], type="l", main="Close", xlab="Date", ylab="Value")
acf(df_INDIAVIX[, c("Close")], lag = 1000, main="Autocorrelation for Close")

library(forecast)
library(fpp2)

len <- nrow(df_INDIAVIX)
df_train <- df_INDIAVIX[0 : (len * 0.85), ]
df_test <- df_INDIAVIX[(len * 0.85) : len, ]

naiv = naive(df_train[, c("Date", "Close")], h = length(df_test[, c("Date", "Close")]), bootstrap = T, level = 0.89)
naivPred <- predict(naiv, df_INDIAVIX[, c("Date", "Close")], type="regression")

naiv
naivPred

par(mfrow=c(1,1))
par(pch=22, col = "black")
plot.new(); plot.window(xlim=c(0, len), ylim = c(min(df_INDIAVIX$Close), max(df_INDIAVIX$Close)))
lines(df_train[, c("Date", "Close")], type="l", col = "black")
lines(df_test[, c("Date", "Close")], type="l", col = "red")
lines(df_INDIAVIX["Date"], naivPred)
