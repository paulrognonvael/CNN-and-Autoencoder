#Flags

#Flags
FLAGS <- flags(
  flag_numeric("filters_nb_start", 32),
  flag_numeric("pooling_size", 2)
)

#train and validation data
train_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = b_size,
  class_mode = "input"
)

validation_datagen <- image_data_generator(rescale = 1/255)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = b_size,
  class_mode = "input"
)


#### Convolutional Encoder
model_enc <- keras_model_sequential() 
model_enc %>%
  layer_conv_2d(filters = FLAGS$filters_nb_start, kernel_size = c(2,2), padding ="same",
                activation = "relu",input_shape = c(64, 64, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2),padding ="same") %>%
  
  layer_conv_2d(filters = FLAGS$filters_nb_start, kernel_size = c(2,2), padding ="same",
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2),padding ="same") %>% 
  
  layer_conv_2d(filters = FLAGS$filters_nb_start/2, kernel_size = c(2,2), padding ="same",
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(FLAGS$pooling_size,FLAGS$pooling_size), padding ="same")

#### Convolutional Decoder 

model_dec <- keras_model_sequential() 
model_dec %>%
  layer_conv_2d(filters = FLAGS$filters_nb_start/2, kernel_size = c(3,3), 
                activation = "relu", padding = "same",
                input_shape = c(64/(2*2*FLAGS$pooling_size), 64/(2*2*FLAGS$pooling_size), FLAGS$filters_nb_start/2))  %>%
  layer_upsampling_2d(size = c(2,2))  %>%
  
  layer_conv_2d(filters = FLAGS$filters_nb_start, kernel_size = c(3,3), 
                activation = "relu", padding = "same")  %>%
  layer_upsampling_2d(size = c(2,2))  %>%
  
  layer_conv_2d(filters = FLAGS$filters_nb_start, kernel_size = c(3,3), 
                activation = "relu", padding = "same")  %>%
  layer_upsampling_2d(size = c(FLAGS$pooling_size,FLAGS$pooling_size))  %>%
  layer_conv_2d(filters = 1, kernel_size = c(1,1), 
                activation = "relu")  
summary(model_dec)

#### Autoencoder 
model_auto<-keras_model_sequential()
model_auto %>%model_enc%>%model_dec
#batch size
b_size <- 50

# set seed for reproductibility
set.seed(42)
tensorflow::tf$random$set_seed(42)

#initialise the model
model_auto %>% compile(
  loss = "mean_squared_error",
  #optimizer = optimizer_rmsprop(),
  optimizer = "adam",
  metrics = c("mean_squared_error")
)

# Fit the model
history_auto <- model_auto %>% fit_generator(
  train_generator,
  steps_per_epoch = 500/b_size,
  epochs = 5,
  validation_data = validation_generator,
  validation_steps = 100/b_size
)
model_auto %>% save_model_hdf5("auto_model.h5")

train_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = b_size,
  class_mode = "input"
)

test_datagen <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = b_size,
  class_mode = "input",
  shuffle = FALSE
)

# From input to encoder - training
predict_enc_train <- model_enc %>% predict_generator(
  train_generator,
  steps = 500/b_size)
dim(predict_enc_train)

# From input to encoder - test
predict_enc_test <- model_enc %>% predict_generator(
  test_generator,
  steps = 100/b_size)
dim(predict_enc_test)

# flat file 
dim(predict_enc_train) <- c(nrow(predict_enc_train),prod(dim(predict_enc_train)[-1]))
dim(predict_enc_test) <- c(nrow(predict_enc_test),prod(dim(predict_enc_test)[-1]))

# Flatten array 
y_radio_train <- train_generator$classes
y_radio_test <- test_generator$classes
save(predict_enc_train,y_radio_train,predict_enc_test, y_radio_train, file=paste0("Conv_Encod_Flat_filter",
                                                                                  FLAGS$filters_nb_start,
                                                                                  "_pool",FLAGS$pooling_size,
                                                                                  ".RData"))



