rm(list=objects())

library(arrow)
library(qgam)
library(dplyr)
library(ranger)
library(tidytable)
library(lubridate)

# Données produites par location.ipynb
train <- read_parquet('data/train2.parquet')
test <- read_parquet('data/test2.parquet')

# Données originales
# train <- read_parquet('data/train.parquet')
# test <- read_parquet('data/test.parquet')

train[train$RatecodeID >= 6, 'RatecodeID'] <- 6.0
test[test$RatecodeID >= 6, 'RatecodeID'] <- 6.0

train <- get_dummies(train, cols=c('VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type'),drop_first = TRUE)
train <- train[,-c('VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type')]

test <- get_dummies(test, cols=c('VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type'))
test <- test[,-c('VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type')]

train$duration <- as.numeric(train$tpep_dropoff_datetime) - as.numeric(train$tpep_pickup_datetime)
test$duration <- as.numeric(test$tpep_dropoff_datetime) - as.numeric(test$tpep_pickup_datetime)

train$log_duration <- log(train$duration + 1)
test$log_duration <- log(abs(test$duration) + 1)

train$long_trip <- 1.0* (train$duration > 70000)
test$long_trip <- 1.0 * (test$duration > 70000)

train$euclidian_distance <- sqrt( (train$DO_location_lon - train$PU_location_lon)^2 + (train$DO_location_lat - train$PU_location_lat)^2 )
test$euclidian_distance <- sqrt( (test$DO_location_lon - test$PU_location_lon)^2 + (test$DO_location_lat - test$PU_location_lat)^2 )

train$speed <- train$trip_distance/(train$duration + 1e-3)
test$speed <- test$trip_distance/(test$duration + 1e-3)

train$log_speed <- log(train$speed)
test$log_speed <- log(abs(test$speed))

train$hour <- hour(train$tpep_dropoff_datetime) + minute(train$tpep_dropoff_datetime) / 60
test$hour <- hour(test$tpep_dropoff_datetime) + minute(test$tpep_dropoff_datetime) / 60

train$week_end <- (wday(train$tpep_dropoff_datetime) >= 6)
test$week_end <- (wday(test$tpep_dropoff_datetime) >= 6)

train$day <- mday(train$tpep_dropoff_datetime)
test$day <- mday(test$tpep_dropoff_datetime)

train$ferie <- 1.0 * ( (train$day == 1) | (train$day == 15) )
test$ferie <- 1.0 * ( (test$day == 1) | (test$day == 15) )

train <- train[,-c("tpep_pickup_datetime", "tpep_dropoff_datetime")]
test <- test[,-c("tpep_pickup_datetime", "tpep_dropoff_datetime")]

# /!\ Cas d'utilisation de train2 et test2
train <- get_dummies(train, cols=c('PU_location','DO_location'))
train <- train[,-c('PU_location', 'DO_location', 'PU_location_outside', 'DO_location_outside')]
test <- get_dummies(test, cols=c('PU_location','DO_location'))
test <- test[,-c('PU_location', 'DO_location', 'PU_location_outside', 'DO_location_outside')]

###################################################################
#### GAM + RF

RFGAM <- function(train0, test1, equation, cov, custom_gterm=NULL){

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
  if (is.null(custom_gterm)) {
    cov <- paste0(c(cov, colnames(terms0)),collapse=' + ')
  }else{
    cov <- paste0(c(cov, colnames(terms0)[custom_gterm]),collapse=' + ')
  }
  equation_rf <-  paste0("res", " ~ ", cov)

  # RF sur les résidus
  rf_gam.res <- ranger::ranger(equation_rf,
                               data = train0_rf,
                               importance =  'permutation',
                               num.trees = 500) # , max.depth = 35, mtry=5)
  rf_gam.forecast <- predict(rf_gam.res, data = train1_rf)$predictions + gam.forecast1
  print(rf_gam.res$variable.importance)

  return(rf_gam.forecast)
}

###########################################################
#### Performance

R2 <- function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

# Cross Validation
CV <- function(data, equation, equation_rf, gterm, K=5){
  cv_R2 <- 0

  ind <- sample(1:K,nrow(data),replace=T)
  for(i in 1:K){

    print(paste0("--> fold ", i))

    train0  <- data[ind != i, ]
    test1   <- data[ind == i, ]

    pred <- RFGAM(train0, test1, equation, equation_rf, gterm)

    # gam.res <- gam(equation, data = train0)
    # pred <- pmax(predict(gam.res, newdata=test1),0)

    cv_R2 <- cv_R2 + R2(test1$tip_amount, pred)
  }
  return(cv_R2/K)
}

# la majorité des variables
equation <- tip_amount ~ passenger_count + s(trip_distance) + s(fare_amount) + s(extra) + mta_tax +
  s(tolls_amount) + improvement_surcharge + congestion_surcharge + Airport_fee +
  te(PU_location_lat, PU_location_lon) + te(DO_location_lat, DO_location_lon) +
  VendorID_2 + RatecodeID_2 + RatecodeID_3 + RatecodeID_4 + RatecodeID_5 + RatecodeID_6 + store_and_fwd_flag_Y +
  payment_type_2 + payment_type_3 + payment_type_4 + s(duration) + s(log_duration) + long_trip + s(euclidian_distance) +
  s(speed) + s(log_speed) + s(hour) + week_end + day +
  PU_location_airport + PU_location_bronx + PU_location_brooklyn + PU_location_manhattan + PU_location_queens + PU_location_staten_island +
  DO_location_airport + DO_location_bronx + DO_location_brooklyn + DO_location_manhattan + DO_location_queens + DO_location_staten_island

gam.res <- gam(equation, data = train)
summary(gam.res)

# best : seulement les variables les plus significatives pour le GAM
equation <- tip_amount ~ s(trip_distance, bs='cr') + s(fare_amount, bs='cr') + s(extra) + mta_tax + s(tolls_amount, bs='cr') + congestion_surcharge +
  te(PU_location_lat, PU_location_lon) + te(DO_location_lat, DO_location_lon) +
  VendorID_2 + RatecodeID_2 + RatecodeID_4 + RatecodeID_6 +
  payment_type_2 + payment_type_3 + payment_type_4 + s(log_duration, bs='cr') + s(euclidian_distance) + s(log_speed, bs='cr') + s(hour, bs='tp') +
  PU_location_airport + PU_location_bronx + PU_location_brooklyn + PU_location_manhattan + PU_location_queens +
  DO_location_airport + DO_location_bronx + DO_location_brooklyn + DO_location_manhattan + DO_location_queens + DO_location_staten_island

# best : la majorité des variables sont utilisées pour prédire les résidus
custom_gterm <- NULL
cov <- c("passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax",
  "tolls_amount", "improvement_surcharge", "congestion_surcharge", "Airport_fee",
  "PU_location_lat", "PU_location_lon", "DO_location_lat", "DO_location_lon",
  "VendorID_2", "RatecodeID_2", "RatecodeID_3", "RatecodeID_4", "RatecodeID_5", "RatecodeID_6", "store_and_fwd_flag_Y",
  "payment_type_2", "payment_type_3", "payment_type_4", "duration", "log_duration", "long_trip", "euclidian_distance",
  "speed", "log_speed", "hour", "week_end", "day",
  "PU_location_airport", "PU_location_bronx", "PU_location_brooklyn", "PU_location_manhattan", "PU_location_queens", "PU_location_staten_island",
  "DO_location_airport", "DO_location_bronx", "DO_location_brooklyn", "DO_location_manhattan", "DO_location_queens", "DO_location_staten_island")

# Les termes avec le plus d'importance dans la random forest
# custom_gterm <- c(21, 22, 27, 28, 29)
# cov <- c("trip_distance","fare_amount","payment_type_2","duration","log_duration","euclidian_distance","speed","log_speed")

CV(train, equation, cov, custom_gterm, K=5)

# overfitting ?
pred <- RFGAM(train, train, equation, cov,custom_gterm)
R2(train$tip_amount, pred)

###########################################################
#### Submit

res <- RFGAM(train, test, equation, cov,custom_gterm)

data.res <- data.frame(row_ID = 1:length(res), tip_amount = res)
write_parquet(data.res,"predictions/GAMRF.parquet")