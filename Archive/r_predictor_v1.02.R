


################################################################################
# Functions
################################################################################

# Function that determines the direction information.
get_dir <- function(dat1){
  # Get the cardinal direction
  convert_heading_to_cardinal <- function(heading) {
    if (is.na(heading)) {
      return("N")  # Default to "N" if heading is missing
    } else if (heading >= 315 || heading < 45) {
      return("N")
    } else if (heading >= 45 && heading < 135) {
      return("E")
    } else if (heading >= 135 && heading < 225) {
      return("S")
    } else if (heading >= 225 && heading < 315) {
      return("W")
    }
  }
  dat1$train_heading <- sapply(dat1$heading, convert_heading_to_cardinal)
  
  # Determine the most common direction for the train.
  # Function to calculate the most common heading for a vector
  get_most_common_heading <- function(heading_vector) {
    heading_freq <- table(heading_vector)
    return(names(which.max(heading_freq)))
  }
  
  # Get the unique train IDs
  unique_train_ids <- unique(dat1$train_id)
  
  # Create a vector to store the average heading for each train ID
  train_heading_avg <- rep(NA, nrow(dat1))
  
  # Loop through each unique train ID and calculate the most common heading
  for (train_id in unique_train_ids) {
    # Get the indices of the rows that belong to the current train_id
    rows <- which(dat1$train_id == train_id)
    
    # Get the most common heading for this train_id
    common_heading <- get_most_common_heading(dat1$train_heading[rows])
    
    # Assign the common heading to the corresponding rows in the 'train_heading_avg' column
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

dat2 <- get_dir(dat1)
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

plot(dat5$poi_dist,dat5$time_to_arrival)

# Perform linear regression
model <- lm(time_to_arrival ~ poi_dist, data = dat5)

# Add the predicted values to the dataframe
dat5$predicted_time_to_arrival <- predict(model, newdata = dat5)

# Calculate the distance between the actual and predicted values
dat5$error <- dat5$time_to_arrival - dat5$predicted_time_to_arrival

# Plot the data points
plot(dat5$poi_dist, dat5$time_to_arrival, 
     xlab = "POI Distance", 
     ylab = "Time to Arrival (minutes)", 
     main = "POI Distance vs Time to Arrival with Regression Line")

# Add the linear regression line
abline(model, col = "red", lwd = 2)  # 'col' sets the color and 'lwd' sets the line width


# Examine dest
p_dest <- ggplot(dat5, aes(x = error, y = dest)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_dest

# Examine service 
p_service <- ggplot(dat5, aes(x = error, y = service)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_service

# Examine current_stop 
p_currentstop <- ggplot(dat5, aes(x = error, y = currentstop)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_currentstop
# Examine train_id 
p_train_id <- ggplot(dat5, aes(x = error, y = train_id)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_train_id
# Examine coinsist 
p_consist <- ggplot(dat5, aes(x = error, y = consist)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_consist
# Examine late 
dat5$late_chr <- as.character(dat5$late)
dat5$late_chr <- factor(dat5$late_chr, levels = sort(unique(as.numeric(dat5$late_chr))), ordered = TRUE)
p_late <- ggplot(dat5, aes(x = error, y = late_chr)) +
  geom_violin() +
  # geom_point() +
  theme_minimal() + 
  xlim(-10,10)
p_late
# Examine SOURCE 
p_SOURCE <- ggplot(dat5, aes(x = error, y = SOURCE)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_SOURCE
# Examine train_speed 
p_train_speed <- ggplot(dat5, aes(x = error, y = train_speed)) +
  # geom_violin() +
  geom_point() +
  theme_minimal()
p_train_speed
# Examine time 
p_time <- ggplot(dat5, aes(x = error, y = time)) +
  # geom_violin() +
  geom_point() +
  theme_minimal()
p_time
# Examine lat 
p_lat <- ggplot(dat5, aes(x = error, y = lat)) +
  # geom_violin() +
  geom_point() +
  theme_minimal()
p_lat
# Examine lon 
p_lon <- ggplot(dat5, aes(x = error, y = lon)) +
  # geom_violin() +
  geom_point() +
  theme_minimal()
p_lon
# Examine heading 
p_heading <- ggplot(dat5, aes(x = error, y = heading)) +
  # geom_violin() +
  geom_point() +
  theme_minimal()
p_heading

p_train_heading <- ggplot(dat5, aes(x = error, y = train_heading)) +
  geom_violin() +
  geom_point() +
  theme_minimal()
p_train_heading
