# DAL ToolBox
# version 1.0.767

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
library(devtools)
load_all("/home/lucas/daltoolbox/")

# Dataset
data(sin_data)

sw_size <- 5
ts <- ts_data(sin_data$y, sw_size)

ts_head(ts)

# Normalization
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)

# Train Test Split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
features <- names(train)

# Create Autoencoder
auto <- cae_encode_decode(5, 3)

auto <- fit(auto, train)

# Testing Autoencoder
print(head(test))
result <- transform(auto, test)
print(head(result))

print(result-test)
train['test_sample'] <- 0
test['test_sample'] <- 1
pred_data <- rbind(train, test)
pred_plot_data <- as.data.frame(transform(auto, pred_data[, features]))
names(pred_plot_data) <- features

ts_df <- rbind(train, test)
ts_df$pred <- 0
ts_df$index <- as.numeric(rownames(ts_df))
pred_plot_data$test_sample <- ts_df$test_sample
pred_plot_data$pred <- 1
pred_plot_data$index <- as.numeric(rownames(pred_plot_data))
rownames(pred_plot_data) <- rownames(ts_df)

plot_data <-rbind(ts_df, pred_plot_data)

ggplot(plot_data, aes(x=index, y=t2, group=pred, colour=pred)) +
  geom_line() +
  ylab('t2')

