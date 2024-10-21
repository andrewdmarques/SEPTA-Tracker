setwd("/media/andrewdmarques/Data01/Personal")


# User input values.
cutoff_count <- 30 # This is the number of data points required for the train to be included. Not enough data points might make it hard for algorythm training. 
poi_train_heading <- 'E'


dat1 <- read.csv('Database/toi.csv')


# format the time
dat1$date_time <- paste0(dat1$date, ' ',dat1$time)
dat1$date_time <- as.POSIXct(dat1$date_time, format = "%Y-%m-%d %H:%M:%S")
dat1$train_heading_avg <- NA
dat1$train_id <- paste0(as.character(dat1$date),'_',as.character(dat1$trainno))
dat1$data_count <- 0
dat1$poi_dist_min <- 100
dat1$poi_depart_time <- NA
dat1$poi_time_to_depart <- NA
dat1$include <- F

# Filter to have just the trains going in the right direction.
dat1 <- subset(dat1,dat1$train_heading_avg == poi_train_heading)

# If there are enough data points then indicate that it should be kept
total_rows <- nrow(dat1)
progress_increment <- total_rows %/% 20 # 5% increment

for (ii in 1:total_rows) {
  # Print progress every 5%
  if (ii %% progress_increment == 0) {
    print(paste0("Processing: ",as.character(ii),"/",as.character(total_rows)," ",as.character(round((ii / total_rows) * 100)),"% complete"), quote = F)
  }
  
  tt <- dat1$train_id[ii]
  t1 <- subset(dat1, dat1$train_id == tt)
  
  # Record the number of data points for this train_id
  dat1$data_count[ii] <- nrow(t1)
  
  # Record the average train_heading
  t2 <- data.frame(table(t1$train_heading))
  t2 <- t2[order(t2$Freq, decreasing = TRUE),]
  t2$Var1 <- as.character(t2$Var1)
  dat1$train_heading_avg[ii] <- t2$Var1[1]
  
  # Record the closest the train got to the poi
  dat1$poi_dist_min[ii] <- min(t1$poi_dist)
  
  # Record the time that the train was closest to the station
  t3 <- subset(t1, t1$poi_dist == min(t1$poi_dist))
  dat1$poi_depart_time[ii] <- t3$time[1]
  
  # Record the time to depart
  dat1$poi_time_to_depart[ii] <- as.numeric(difftime(t3$date_time[1], dat1$date_time[ii], units = "mins"))
}

print('Processing complete', quote = F)


# Filter to have the trains with enough data for predictions.
dat2 <- subset(dat1,dat1$data_count >= cutoff_count)

# Filter to have just the trains going in the right direction.
dat3 <- subset(dat2,dat2$train_heading_avg == poi_train_heading)

# Visualize the time to the station.
dat4 <- dat3
dat4$poi_time_to_depart[dat4$poi_time_to_depart > 20 ] <- 20
dat4$poi_time_to_depart[dat4$poi_time_to_depart < -20] <- -20
ggplot(dat4, aes(x = lon, y = lat, color = poi_time_to_depart)) +
  geom_point(shape = 16, size = 3) + # shape 16 is a solid circle
  scale_color_gradient2(low = "blue", mid = "black", high = "red", midpoint = 0) + # Set the color gradient with limits
  labs(title = "Scatter Plot of Latitude vs Longitude",
       x = "Longitude",
       y = "Latitude",
       color = "Time to Depart") +
  theme_minimal() +
  theme(legend.position = "right") # Optional: move the legend to the right



