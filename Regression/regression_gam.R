rm(list=objects())

library(arrow)
library(qgam)
library(dplyr)
library(ranger)

train <- read_parquet('data/modified/train_modified.parquet')[c(-1,-2)]
test <- read_parquet('data/modified/test_modified.parquet')[c(-1,-2)]

###################################################################
#### GAM + RF

RFGAM <- function(train0, test1, equation, cov){

  # GAM
  gam.res <- gam(equation, data = train0)                     # GAM sur train
  gam.forecast0 <- predict(gam.res, newdata=train0)           # Prédiction sur train
  gam.forecast1 <- predict(gam.res, newdata=test1)            # Prédiction sur test

  # Calcul des résidus sur train
  terms0 <- predict(gam.res, newdata=train0, type='terms')
  colnames(terms0) <- paste0("gterms_", c(1:ncol(terms0)))
  train0_rf <- data.frame(train0, terms0)                     # Estimation des résidus de train
  train0_rf$res <- train0$tip_amount - gam.forecast0          # Résidus de train

  # Calcul des résidus sur test
  terms1 <- predict(gam.res, newdata=test1, type='terms')
  colnames(terms1) <- paste0("gterms_", c(1:ncol(terms1)))
  train1_rf <- data.frame(test1, terms1)                      # Estimation des résidus de test

  # Equation pour les résidus
  cov <- paste0(c(cov, colnames(terms0)),collapse=' + ')
  equation_rf <-  paste0("res", " ~ ", cov)

  # RF sur les résidus
  rf_gam.res <- ranger::ranger(equation_rf, data = train0_rf, importance =  'permutation')
  rf_gam.forecast <- predict(rf_gam.res, data = train1_rf)$predictions + gam.forecast1

  return(rf_gam.forecast)
}

###########################################################
#### Performance

R2 <- function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

# Cross Validation
CV <- function(data, equation, equation_rf, iter=5){
  cv_R2 <- 0
  for(i in 1:iter){
    sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
    train0  <- data[sample, ]
    test1   <- data[!sample, ]

    pred <- RFGAM(train0, test1, equation, equation_rf)

    # gam.res <- gam(equation, data = train0)
    # pred <- predict(gam.res, newdata=test1)

    # print(summary(gam.res))

    cv_R2 <- cv_R2 + R2(test1$tip_amount, pred)
  }
  return(cv_R2/iter)
}

equation <- tip_amount ~ passenger_count + trip_distance + fare_amount + extra + mta_tax + tolls_amount + improvement_surcharge +
  congestion_surcharge + Airport_fee + PU_location_lat + PU_location_lon + DO_location_lat + DO_location_lon + VendorID_2 +
  RatecodeID_2.0 + RatecodeID_3.0 + RatecodeID_4.0 + RatecodeID_5.0 + RatecodeID_6.0

equation <- tip_amount ~ passenger_count + s(trip_distance) + fare_amount + extra + tolls_amount +
  congestion_surcharge + Airport_fee + te(PU_location_lat + PU_location_lon) + te(DO_location_lat + DO_location_lon) + VendorID_2 +
  RatecodeID_2.0 + RatecodeID_3.0 + RatecodeID_4.0 + RatecodeID_5.0 + RatecodeID_6.0

cov <- c("trip_distance","extra")

CV(train, equation, cov, iter=3)

###########################################################
#### Submit

res <- RFGAM(train, test, equation, cov)

data.res <- data.frame(row_ID = 1:length(res), tip_amount = res)
write_parquet(data.res,"predictions/GAMRF.parquet")