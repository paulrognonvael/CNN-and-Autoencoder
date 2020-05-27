set.seed(43)
tensorflow::tf$random$set_seed(43)

train_dir<-"rxtorax/train"
validation_dir<-"rxtorax/validation"
test_dir<-"rxtorax/test"

#Flags
FLAGS <- flags(
  flag_numeric("batch_size", 25)
)


# Normalize images.
train_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = FLAGS$batch_size,
  class_mode = "binary"
)

validation_datagen <- image_data_generator(rescale = 1/255)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = FLAGS$batch_size,
  class_mode = "binary"
)

test_datagen <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  color_mode = "grayscale",
  target_size = c(64, 64),
  batch_size = FLAGS$batch_size,
  class_mode = "binary",
  classes = c("effusion","normal"),
  shuffle = FALSE
)
# Now we have the images in the required format: 64x64 with a unique channel

# sample output of one of the generators we just defined 
batch <- generator_next(train_generator)
str(batch)


# initialise model
model <- keras_model_sequential() %>%
  # first convvolutional hidden layer and max pooling
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = "relu",input_shape = c(64, 64, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # second convvolutional hidden layer and max pooling
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate=0.4) %>%
  # Outputs from dense layer are projected onto output layer
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# Fit the model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = round(500/FLAGS$batch_size)+1,
  epochs = 13,
  validation_data = validation_generator,
  validation_steps = round(100/FLAGS$batch_size)+1
)
