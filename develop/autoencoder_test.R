# DAL ToolBox
# version 1.0.767

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
library(devtools)
load_all("/home/lucas/daltoolbox/R/")

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

# Create Autoencoder
auto <- vae_encode_decode(5, 3)

auto <- fit.vae_encode_decode(auto, train)

# Testing Autoencoder
print(head(test))
result <- transform.vae_encode_decode(auto, test)
print(head(result))

print(result-test)
