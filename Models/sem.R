library(lavaan)
library(qgraph)
library(semPlot)
data("PoliticalDemocracy")
PD <- PoliticalDemocracy

data1 <- read.csv("indicators_new1.csv")
data2 <- read.csv("indicators_new2.csv")
data3 <- read.csv("Turning Points_0629_2.csv", nrows = 106)
Is.War.Confilct <- data3$War.conflict.[2:106]

data_use <- cbind(data2, data1$Is.Turning.Point, Is.War.Confilct)
colnames(data_use)[ncol(data_use)-1] <- "Is.Turning.Point"

model <- '
        # measurement model
          Turning.Point =~ Unemployment.Rate+Military.Spending+GDP+Deficit+Oil.Production
          Tech =~ Unemployment.Rate+Military.Spending+GDP+Deficit+Oil.Production+Oil.Price
          War.Conflict =~ Unemployment.Rate+Military.Spending+GDP+Deficit+Oil.Production
        # regression
          War.Conflict ~ Turning.Point+Tech
        # Residual Correlation
          Unemployment.Rate ~ GDP+Deficit
          Military.Spending ~ Deficit
          GDP ~ Unemployment.Rate
          Deficit ~ Unemployment.Rate+Military.Spending
'

fit <- sem(model, data_use)


sink("sem_output.txt")
summary(fit)
sink()
