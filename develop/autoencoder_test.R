# DAL ToolBox
# version 1.0.767

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
library(devtools)
load_all("/home/lucas/daltoolbox/")
library(ggpubr)

# Dataset
#data(sin_data)

sw_size <- 5

ts <- read.csv('/home/lucas/daltoolbox/develop/data/weather.csv', row.names=2)
ts[,'X'] <- NULL

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
auto <- vae_encode_decode(length(ts), 2)
ae_type <- 'decoder'

auto <- fit(auto, train)

# Testing Autoencoder
result <- transform(auto, test)

train['test_sample'] <- 0
test['test_sample'] <- 1
pred_data <- rbind(train, test)
rec_data <- as.data.frame(transform(auto, pred_data[, features]))


ts_df <- rbind(train, test)
ts_df$index <- as.numeric(rownames(ts_df))

if (ae_type == 'encoder'){
  output_features <- names(rec_data)
  plot_data <-cbind(ts_df, rec_data[output_features])
  
  plot_features <- c(features, output_features)
  plotList <- lapply(
    plot_features,
    function(key) {
      plt <- ggplot(plot_data, aes(x=index, y=eval(parse(text=key)))) +
        geom_line() +
        xlab('') +
        ylab(key) + 
        theme_classic()
      
      plt
    }
  )
  
  ggarrange(
    plotlist=plotList,
    align='v',
    ncol=1, nrow=length(plot_features))
}else{
  pred_plot_data <- rec_data
  names(pred_plot_data) <- features
  
  output_features <- lapply(
    features,
    function(key) {
      new_string <- paste0(key, "_rec")      
      
      new_string
    }
  )
  
  ts_df$pred <- 0
  pred_plot_data$test_sample <- ts_df$test_sample
  pred_plot_data$pred <- 1
  pred_plot_data$index <- as.numeric(rownames(pred_plot_data))
  rownames(pred_plot_data) <- rownames(ts_df)
  names(pred_plot_data) <- c(output_features, c('test_sample', 'pred', 'index'))
  
  plot_data <- pred_plot_data
  
  plot_features <- output_features
  plotList <- lapply(
    output_features,
    function(key) {
      plt <- ggplot(plot_data, aes(x=index, y=eval(parse(text=key)))) +
        geom_line() +
        xlab('') +
        ylab(key) +
        theme_classic()
      
      plt
    }
  )
  
  ggarrange(
    plotlist=plotList,
    align='v',
    ncol=1, nrow=length(features))
}
