

################################################################################
# Functions
################################################################################

# Function that determines the direction information and updates missing 'train_heading'
get_dir <- function(dat1, poi_train_heading) {
  
  # Function to convert heading to cardinal direction based on specific rules
  convert_heading_to_cardinal <- function(heading, poi_heading) {
    if (is.na(heading)) {
      return(NA)  # Keep NA if heading is missing
    }
    
    if (poi_heading %in% c("E", "W")) {
      if (heading >= 0 && heading <= 180) {
        return("E")
      } else if (heading > 180 && heading <= 360) {
        return("W")
      }
    } else if (poi_heading %in% c("N", "S")) {
      if ((heading >= 0 && heading <= 90) || (heading > 270 && heading <= 360)) {
        return("N")
      } else if (heading > 90 && heading <= 270) {
        return("S")
      }
    }
    
    return(NA)  # Return NA if no condition is satisfied
  }
  
  # Apply the logic to fill missing 'train_heading' values
  for (idx in 1:nrow(dat1)) {
    if (is.na(dat1$train_heading[idx])) {
      dat1$train_heading[idx] <- convert_heading_to_cardinal(dat1$heading[idx], poi_train_heading)
    }
  }
  
  # Function to get the most common heading for each train_id
  get_most_common_heading <- function(heading_vector) {
    heading_freq <- table(heading_vector)
    return(names(which.max(heading_freq)))
  }
  
  # Get the unique train IDs and calculate the most common heading for each train ID
  unique_train_ids <- unique(dat1$train_id)
  train_heading_avg <- rep(NA, nrow(dat1))
  
  for (train_id in unique_train_ids) {
    rows <- which(dat1$train_id == train_id)
    common_heading <- get_most_common_heading(dat1$train_heading[rows])
    train_heading_avg[rows] <- common_heading
  }
  
  # Add the new column to the dataframe
  dat1$train_heading_avg <- train_heading_avg
  
  return(dat1)
}


################################################################################
# Run
################################################################################

# Read in the specified data files.
con1 <- read.csv('config.csv')
# file_toi <- subset(con1,con1$variable == 'toi_file')$value[1]
# Assign all the vairbales fomr the con1 data frame.
mapply(assign, con1$variable, con1$value, MoreArgs = list(envir = .GlobalEnv))

toi_file <- '/media/andrewdmarques/Data01/Personal/Database/toi_14.csv'
dat1 <- read.csv(toi_file)
dat1$date_time <- as.POSIXct(dat1$date_time, format = "%Y-%m-%d %H:%M:%S")
dat1$time <- as.POSIXct(dat1$time, format = "%H:%M:%S")
# Subset to just have the trains of interest.

dat2 <- get_dir(dat1,poi_train_heading)
dat2 <- dat2[grepl(pattern = "media", x = dat2$line, ignore.case = TRUE), ]
dat3 <- dat2[grepl(pattern = poi_train_heading, x = dat2$train_heading_avg, ignore.case = TRUE), ]



# Remove any that are beyond 8 miles away, these tend to distort the data set
dat3 <- subset(dat3,dat3$poi_dist < 8)

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


# Remove any trains that were just sitting for more than 1 hour, this will skew the data
t1 <- subset(dat4,dat4$time_to_arrival >= 60)
t2 <- unique(t1$train_id)
# Remove row in data frame by column value (using a list to compare against).
dat5 <- subset(dat4,! dat4$train_id %in% c(t2))

# Only look at the trains that have not yet arrived.
dat5 <- subset(dat5,dat5$time_to_arrival >= 0)

# Keep columns by name
dat6 <- dat5[ , which(names(dat5) %in% c('lat','lon','time_to_arrival','train_id','poi_dist'))]

# Model with speed unbounded ###################################
# Attempting to make a model that captures the train speed also
dat7 <- dat5
dat7$train_speed_t <- dat7$train_speed
# Replace values in 'train_speed_t' that are below 10 with 20
dat7$train_speed_t[dat7$train_speed_t < 10] <- 20
dat7$train_speed_t[dat7$train_speed_t > 20] <- 20

model2 <- lm(time_to_arrival ~ train_speed_t + poi_dist, data = dat7)

# View the summary of the linear model
summary(model2)
dat7$pred <- predict(model2, newdata = dat7) 
dat7$error <- dat7$time_to_arrival - dat7$pred
t1 <- subset(dat7,dat7$error < 4)
t1 <- subset(t1,t1$error > -1)
err_mod2 <- nrow(t1)/nrow(dat7)

plot(dat7$poi_dist, dat7$time_to_arrival, 
     xlab = "POI Distance", 
     ylab = "Time to Arrival (minutes)", 
     main = "POI Distance vs Time to Arrival with Regression Line",
     xlim = c(0,8))

# Add the linear regression line
points(dat7$poi_dist, dat7$pred, col = "red", pch = ".")