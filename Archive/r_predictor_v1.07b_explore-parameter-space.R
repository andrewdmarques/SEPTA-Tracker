# Create a vector of lower bounds to iterate through
lower_bounds <- c(0, 1, 2, 5, 10, 15, 20, 30)

# Initialize a dataframe to store the results
results <- data.frame(lower_bound = numeric(), error_ratio = numeric())

# Function to calculate model and error for a given lower bound
calculate_model_and_error <- function(lower_bound) {
  # Copy the dataframe
  dat7_mod <- dat7
  
  # Replace values in 'train_speed_t' that are below lower_bound with 20
  dat7_mod$train_speed_t <- ifelse(dat7_mod$train_speed < lower_bound, 20, 
                                   ifelse(dat7_mod$train_speed > 20, 20, dat7_mod$train_speed))
  
  # Fit the linear model
  model <- lm(time_to_arrival ~ train_speed_t + poi_dist, data = dat7_mod)
  
  # Make predictions and calculate the error term
  dat7_mod$pred <- predict(model, newdata = dat7_mod)
  dat7_mod$error <- dat7_mod$time_to_arrival - dat7_mod$pred
  
  # Subset data to calculate error ratio
  t1 <- subset(dat7_mod, dat7_mod$error < 4 & dat7_mod$error > -1)
  error_ratio <- nrow(t1) / nrow(dat7_mod)
  
  # Save the error_ratio in results dataframe
  results <<- rbind(results, data.frame(lower_bound = lower_bound, error_ratio = error_ratio))
  
  # Plot the data points and the regression line
  plot(dat7_mod$poi_dist, dat7_mod$time_to_arrival, 
       xlab = "POI Distance", 
       ylab = "Time to Arrival (minutes)", 
       main = paste("POI Distance vs Time to Arrival with lower_bound =", lower_bound),
       xlim = c(0, 8))
  
  # Add the regression line
  points(dat7_mod$poi_dist, dat7_mod$pred, col = "red", pch = ".")
  
  # Save the plot
  filename <- paste0("plot_lower_bound_", lower_bound, ".png")
  dev.copy(png, filename)
  dev.off()
}

# Iterate over the lower_bounds and apply the function
for (bound in lower_bounds) {
  calculate_model_and_error(bound)
}

# View the results
print(results)
