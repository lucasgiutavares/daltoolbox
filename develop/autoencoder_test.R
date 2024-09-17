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
#ts <- ts_data(sin_data$y, sw_size)

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
auto <- cae_encode(length(ts), 3)
ae_type <- 'encoder'

auto <- fit(auto, train)

# Testing Autoencoder
print(head(test))
result <- transform(auto, test)
print(head(result))

print(result-test)
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
    #heights=c(6, 6, 6, 14),
    #widths=c(8, 8, 8, 18),
    align='v',
    ncol=1, nrow=length(plot_features))
}else{
  pred_plot_data <- rec_data
  names(pred_plot_data) <- features
  
  ts_df$pred <- 0
  pred_plot_data$test_sample <- ts_df$test_sample
  pred_plot_data$pred <- 1
  pred_plot_data$index <- as.numeric(rownames(pred_plot_data))
  rownames(pred_plot_data) <- rownames(ts_df)
  
  plot_data <- rbind(ts_df, pred_plot_data)
  
  plot_features <- c(features, output_features)
  plotList <- lapply(
    features,
    function(key) {
      plt <- ggplot(plot_data, aes(x=index, y=eval(parse(text=key)), group=pred, colour=pred)) +
        geom_line() +
        xlab('') +
        ylab(key) + 
        theme_classic()
      
      plt
    }
  )
  
  ggarrange(
    plotlist=plotList,
    #heights=c(6, 6, 6, 14),
    #widths=c(8, 8, 8, 18),
    align='v',
    ncol=1, nrow=length(features))
}
