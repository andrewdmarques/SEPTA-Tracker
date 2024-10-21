# Read in the specified data files.
con1 <- read.csv('config.csv')
# file_toi <- subset(con1,con1$variable == 'toi_file')$value[1]
# Assign all the vairbales fomr the con1 data frame.
mapply(assign, con1$variable, con1$value, MoreArgs = list(envir = .GlobalEnv))


dat1 <- read.csv(toi_file)
dat1$date_time <- as.POSIXct(dat1$date_time, format = "%Y-%m-%d %H:%M:%S")
dat1$time <- strptime(dat1$time, format = "%H:%M:%S")
# Subset to just ahve the trains of interest.
dat2 <- dat1[grepl(pattern = "media", x = dat1$line, ignore.case = TRUE), ]
dat3 <- dat2[grepl(pattern = poi_train_heading, x = dat2$train_heading_avg, ignore.case = TRUE), ]

# Remove any that are beyond 8 miles away, these tend to distort the data set
dat3 <- subset(dat3,dat4$poi_dist < 8)

# Estimate when the time of arrival for the train would be.
dat4 <- dat3
dat4$time_arrival <- dat4$date_time
dat4$time_to_arrival <- 0
for(ii in 1:nrow(dat4)){
  t1 <- dat4$train_id[ii]
  t2 <- subset(dat4,dat4$train_id == t1)
  t3 <- subset(t2,t2$poi_dist == min(t2$poi_dist))
  dat4$time_arrival[ii] <- t3$date_time[1]
  
  time_diff <- difftime(t3$date_time[1], dat4$date_time[ii], units = "secs")
  dat4$time_to_arrival[ii] <- as.numeric(time_diff) / 60  # Convert seconds to minutes
  
  # dat4$time_to_arrival[ii] <- t3$date_time[1] - dat4$date_time[ii]
}

# Only look at the trains that have not yet arrived.
dat5 <- subset(dat4,dat4$time_to_arrival >= 0)

# Keep columns by name
dat6 <- dat5[ , which(names(dat5) %in% c('lat','lon','time_to_arrival','train_id','poi_dist'))]

plot(dat6$poi_dist,dat6$time_to_arrival)

# Perform linear regression
model <- lm(time_to_arrival ~ poi_dist, data = dat6)

# Add the predicted values to the dataframe
dat6$predicted_time_to_arrival <- predict(model, newdata = dat6)

# Calculate the distance between the actual and predicted values
dat6$dist_from_pred <- dat6$time_to_arrival - dat6$predicted_time_to_arrival

# Plot the data points
plot(dat6$poi_dist, dat6$time_to_arrival, 
     xlab = "POI Distance", 
     ylab = "Time to Arrival (minutes)", 
     main = "POI Distance vs Time to Arrival with Regression Line")

# Add the linear regression line
abline(model, col = "red", lwd = 2)  # 'col' sets the color and 'lwd' sets the line width

# Optionally print the first few rows to check the new column
head(dat6)
