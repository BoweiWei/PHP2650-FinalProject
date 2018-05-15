df <- read.csv("default of credit card clients.csv", header = TRUE, sep = ",")
df$SEX <- as.factor(df$SEX)
df$EDUCATION <- as.factor(df$EDUCATION)
df$MARRIAGE <- as.factor(df$MARRIAGE)
df$credhist <- ifelse((df$PAY_0 | df$PAY_2 | df$PAY_3 | df$PAY_4 | df$PAY_5 | df$PAY_6) > 0, "Bad", "Good") # check
# whether the customer has the payment delay history. if all clear, set as Good. Equal to 0 means they want to pay the
# bill although they cannot finish them all right now. We conclude it as good credit still. 
df$PAY_0 <- as.factor(df$PAY_0)
df$PAY_2 <- as.factor(df$PAY_2)
df$PAY_3 <- as.factor(df$PAY_3)
df$PAY_4 <- as.factor(df$PAY_4)
df$PAY_5 <- as.factor(df$PAY_5)
df$PAY_6 <- as.factor(df$PAY_6)
df$default.payment.next.month <- as.factor(default.payment.next.month)

levels(df)
is.na.data.frame(df)
