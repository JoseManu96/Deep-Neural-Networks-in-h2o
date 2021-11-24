### DNN :####

# Incluimos el paquete:
library(h2o)

# Inicializamos:
h2o.init(nthreads = -1) 

# Cargamos los conjuntos de entrenamiento, prueba y validación:
train <-  read.csv("C:/Users/Jose Fuentes/Desktop/MNISTtrain_40000.csv")
test<-  read.csv("C:/Users/Jose Fuentes/Desktop/MNISTtest_9000.csv")
val<-  read.csv("C:/Users/Jose Fuentes/Desktop/MNISTvalidate_11000.csv")

# Los convertimos en elementos de h2o:
train=as.h2o(train)
test=as.h2o(test)

# La última columna se deja como tipo factor:
y = "C785" 
x = setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

# Comprobamos las dimensiones:
dim(train_h2o)
dim(test_h2o)

# Aplicamos Grid Search:

# Especificamos los hiperparámetros:
l1_opt=c(1e-3,1e-4,1e-5,1e-6)
l2_opt=c(1e-4,1e-5,1e-6,0)
hidden_opt <- list(c(200,200,200), c(200,200,200,200))
idr=c(0.1,0.2)
ep=c(10)
(hyper_params <- list(hidden = hidden_opt, l1 = l1_opt, l2= l2_opt, epochs=ep,
                      input_dropout_ratio=idr))

# Ejecutamos el h2o.grid:
grid_search1 = h2o.grid(algorithm="deeplearning",
                       grid_id = "grid_search1",
                       hyper_params = hyper_params,
                       x = x,
                       y = y,
                       activation="RectifierWithDropout", # Fijamos esta función de activación puesto que es la que se emplea el el trato de imagenes.
                       training_frame = train,
                       validation_frame = test)

grid_search1 # A partir del grid anterior optamos por usar como valores de 
# l2 y l1, 0 y 1e-5, respectivamente.

# Hidden Dropout rario lo fijamos como 0.2 dado que la documentacion sugiere este,
# valor, así como 0.1, sin embargo para 0.1 los resultados no fueron buenos.

# Definimos la siguiente función para variar las capas ocultas:
h_g=function(x){
  return(c(sample(c(256,512,1024,2048),c(3,4),replace = T)))
} 

# Definimos ahora los nuevos hiperparámetos:
(hidden_opt = lapply(1:20, function(x)sample(c(256,512,1024,2048),c(3,4), replace=TRUE)))
activation_opt <- c(  "RectifierWithDropout")
l2_opt <- c(0)
ep_opta=10
hyper_params <- list(activation = "RectifierWithDropout",hidden=hidden_opt, l1 = 1e-5, l2 = l2_opt,input_dropout_ratio=0.2,epochs=ep_opta)

# Definimos la búsqueda como aleatoria pero para un máximo de 10 modelos.
search_criteria <- list(strategy = "RandomDiscrete", max_models=10)

# Ejecutamos el segundo grid:
grid_search2 <- h2o.grid("deeplearning", x = x, y = y, distribution="multinomial",
                     grid_id = "grid_search2",
                     training_frame = train,
                     validation_frame = test,
                     seed = 1,
                     hyper_params = hyper_params,stopping_metric="logloss",
                     search_criteria = search_criteria)

# Guardamos el grid:
#gridb_path <- h2o.saveGrid(grid_id =grid_search1@grid_id, grid_directory = "C:/Users/Jose Fuentes/Desktop/Modelos")
#print(gridb_path)

# Para cargarlo empleamos la siguiente función:
#grid <- h2o.loadGrid("C:/Users/Jose Fuentes/Desktop/Modelos/grid_search1")
#grid

# Ordenamos los modelos según el accuracy:
dl_gridperf <- h2o.getGrid(grid_id = "grid_search1", 
                           sort_by = "accuracy", 
                           decreasing = TRUE)
print(dl_gridperf) # Imprimimos

# Definimos listas para guardar nuestros resutados:
mfit5b=list()
mprfTrn5b=list()
mprfTst5b=list()

mfit5b <- lapply(dl_gridperf@model_ids, function(id) { h2o.getModel(id)})
grid@model_ids

# Guardamos el performance de los 10 modelos:
system.time(
  for(i in 1:10){
    mprfTrn5b[[i]] <- h2o.performance(mfit5b[[i]], newdata = train1)
    mprfTst5b[[i]] <- h2o.performance(mfit5b[[i]], newdata = test1)
  })


dev.off()
par(mfrow = c(2,5), mai = c(.8,.8,.8,.8))

# Graficamos la history de los 10 modelos: 
for(i in 1:10){
  plot(mfit5b[[i]],metric = "classification_error",cex=.7)
}

# Guardamos los resultados en una tabla:
tab3b=matrix(NA, nrow=10, ncol=10)
colnames(tab3b)=c("model","RunTimeMins","layers","epochs","l1","InDropoutRat",
                  "loglss_trn","loglss_tst","Err_trn","Err_tst")

for(i in 1:10){
  tab3b[i,1]=mfit5b[[i]]@model_id
  tab3b[i,2]=round(mfit5b[[i]]@model$run_time/60000, digits=2)
  tab3b[i,3]=paste(mfit5b[[i]]@allparameters$hidden, collapse="-")
  tab3b[i,4]=round(mfit5b[[i]]@allparameters$epochs, digits=2)
  tab3b[i,5]=mfit5b[[i]]@allparameters$l1
  tab3b[i,6]=mfit5b[[i]]@allparameters$input_dropout_ratio 
  tab3b[i,7]=round(h2o.logloss(mprfTrn5b[[i]]), digits=4)
  tab3b[i,8]=round(h2o.logloss(mprfTst5b[[i]]), digits=4)
  tab3b[i,9]=round(mprfTrn5b[[i]]@metrics$cm$table$Error[11], digits=4)
  tab3b[i,10]=round(mprfTst5b[[i]]@metrics$cm$table$Error[11], digits=4)
}
tab3b

# Graficamos los resultados de los 10 modelos anteriores:
library(ggplot2)
par(mfrow = c(1,1), mai = c(.8,.8,.8,.8))
matplot(1:10, 
        cbind(tab3b[,7], 
              tab3b[,8],
              tab3b[,9], 
              tab3b[,10]
        ),
        pch=19, 
        col=c("lightblue", "orange","blue","red"),
        type="b",ylab="Errors/logloss", xlab="Modelos",
        main="Malla aleatoria",xaxt="n",yaxt="n")
axis(1, at=c(1:10), labels=c(10,5,6,4,2,9,3,8,1,7),
     tck = 1,lty=2,col="gray")
axis(1, at=c(1:10), labels=c(10,5,6,4,2,9,3,8,1,7),
     tick = 1)
axis(2, at=1:35/100, labels=1:35/100,tck = 1,lty=2,col="gray")
axis(2, at=1:35/100, labels=1:35/100,tick = 1)
box()
legend("topright", legend=c("Logloss_train", "Logloss_test",
                            "Error_train", "Error_test"), pch=19, cex=.7,
       col=c("lightblue", "orange","blue","red"))

# Guardamos los resultados individuales de las estadísticas de interés
library(xtable)
print(xtable(tab3b), include.rownames = FALSE)

# Basado en los 2 mejores modelos respecto al accuracy, pasamos a aumentar el 
# número de épocas.

# Primero escalamos las X's, dado que nos ayuda a obtner mejores resultados en 
# el tratado de imágenes.
train1=train
train1[,1:784]=train1[,1:784]/255
test1=test
test1[,1:784]=test1[,1:784]/255
test1[,785]

# Para nuestra nueva arquitectura usamos los hyperparameters que mejores resultados
# mostraron hasta el momento, pero aumentamos las epochs y especificamos logloss
# en la metrica de parada:
fit.dl1 <- h2o.deeplearning(model_id = "fit.dl1",distribution = 'multinomial',
                             x = x,  # predictors
                             y = y,   # label
                             training_frame = train1, 
                             validation_frame = test1,
                             activation = "RectifierWithDropout", 
                             input_dropout_ratio = 0.2,
                             hidden = c(1024,1024,2048)
                             ,l1=1e-5,l2=0, 
                             epochs = 100,stopping_metric = "logloss",
                             seed = 1
)

# Comprobamos los resultados:
fit.dl1@model
pred1=h2o.performance(fit.dl1,newdata = test1)
pred1@metrics$cm$table$Error # Accuracy: 98.22

# Guardamos el modelo:
#model_path <- h2o.saveModel(fit.dl1, path = "C:/Users/Jose Fuentes/Desktop")
#print(model_path)

# Esto se usa para cargarlo:
#t=h2o.loadModel(path = "C:/Users/Jose Fuentes/Desktop/fit.model1")
#h2o.performance(t,newdata = test1)

# Para el segundo mejor modelo hacemos lo mismo:
fit.dl2<- h2o.deeplearning( x = x, y = y, distribution="multinomial",
                               model_id = "fit.model2",
                               training_frame = train1,
                               validation_frame = test1,
                               seed = 1,
                               activation = "RectifierWithDropout",hidden=c(2048,2048,1024),
                               l1 = 1e-5, l2 = 0,input_dropout_ratio=0.2,epochs=100
                               ,stopping_metric="logloss")

# Comprobamos los resultados:
fit.dl2@model
pred2=h2o.performance(fit.dl2,newdata = test1)
pred2@metrics$cm$table$Error # Accuracy: 98.25

# A partir del modelo de los 2 anteriores que mostrara mayor accuracy, decidimos
# tomarlo como el mejor y pasamos a hacer cross-validation con 2 folds.

fit.dl3 <- h2o.deeplearning( x = x, y = y, distribution="multinomial",
                               model_id = "fit.model3",
                               training_frame = train1,
                               validation_frame = test1,
                               seed = 1,
                               activation = "RectifierWithDropout",hidden=c(2048,2048,1024),
                               l1 = 1e-5, l2 = 0,input_dropout_ratio=0.2,epochs=100,nfolds = 2
                               ,stopping_metric="logloss")

# Lo guardamos:
#model_path <- h2o.saveModel(fit.dl3, path = "C:/Users/Jose Fuentes/Desktop")
#print(model_path)

# Comprobamos el accuracy, asi como otras metricas de interes:
fit.dl3@model
pred=h2o.performance(fit.dl3,newdata = train1)
pred@metrics$cm$table$Error # Accuracy: 98.3667 (test);  99.935 (train)  

# Errores del train por clase y global:
# 0.0002546473 0.0002225684 0.0005033979 0.0000000000 0.0005113782 0.0008234971 0.0010180708 0.0012045290
# 0.0002597403 0.0017395626 0.0006500000


# Errores del test por clase y global:
# 0.004494382 0.007850834 0.013186813 0.033934252 0.013745704 0.011180124 0.015503876 0.017316017
# 0.022624434 0.023557126 0.016333333

# Cross-Validation Metrics Summary: 
                             #mean           sd  cv_1_valid  cv_2_valid
#accuracy                 0.97361994 0.0041999966   0.9706501   0.9765898
#auc                             NaN          0.0         NaN         NaN
#err                     0.026380049 0.0041999966 0.029349895 0.023410203
#err_count                     527.5     82.73149       586.0       469.0
#logloss                  0.11740296  0.007062476 0.122396894  0.11240904
#max_per_class_error       0.0465174  0.011957338 0.054972515 0.038062282
#mean_per_class_accuracy  0.97341055   0.00420489   0.9704373  0.97638386
#mean_per_class_error    0.026589433   0.00420489 0.029562738 0.023616126
#mse                      0.02258594 0.0032485218  0.02488299 0.020288888
#pr_auc                          NaN          0.0         NaN         NaN
#r2                        0.9972939 3.9011156E-4  0.99701804  0.99756974
#rmse                     0.15009125  0.010821822  0.15774344  0.14243907

# Predecimos los valores del validation test:
val=as.h2o(val)
val1=val
val1[,1:784]=val1[,1:784]/255

# Guardamos los valores:
pred=predict(fit.dl3,newdata = val1)
print(pred[,1])
val_pred=as.data.frame(pred[,1])
write.csv(val_pred, file="Proyecto_FuentesPatarroyo_DNN_pred.csv", row.names=F)

#### Convolutional Neural Network:####
library(rlang)
library(dplyr)

install.packages("tensorflow")
install.packages("keras")
install.packages("dplyr")
install.packages("yardstick")

library(dplyr)
library(keras)
library(tensorflow)
library(yardstick)
library(reticulate)

train_mnist <- read.csv("C:/Users/Jose Fuentes/Desktop/MNISTtrain_40000.csv")
test_mnist <- read.csv("C:/Users/Jose Fuentes/Desktop/MNISTtest_9000.csv")
val_mnist <- read.csv("C:/Users/Jose Fuentes/Desktop/MNISTvalidate_11000.csv")
head(train_mnist)

# Cargamos los datos
data_train <- train_mnist
data_test <- test_mnist
data_val <- val_mnist

train_x <- data_train %>% 
  select(-785) %>% 
  as.matrix() / 255

test_x <- data_test %>% 
  select(-785) %>% 
  as.matrix() / 255

val_x <- data_val %>% 
  as.matrix() / 255

train_y <- to_categorical(data_train$C785, num_classes = 10) # One hot encoding

head(train_x)

# Reajustamos las dimensiones de las matrices:
train_x_keras <- array_reshape(train_x, dim = c(nrow(train_x), 28, 28, 1))

test_x_keras <- array_reshape(test_x, dim = c(nrow(test_x), 28, 28, 1))

val_x_keras <- array_reshape(val_x, dim = c(nrow(val_x), 28, 28, 1))

dim(val_x_keras)

tensorflow::tf$random$set_seed(123)

# Ejecutamos la arquitectura:
model <- keras_model_sequential(name = "CNN_Model") %>% 
  
  layer_conv_2d(filters = 32, 
                kernel_size = c(4,4), 
                padding = "same", activation = "relu",
                input_shape = c(28, 28, 1)
  ) %>% 
  
  layer_max_pooling_2d(pool_size = c(3,3)) %>% 
  
  layer_conv_2d(filters = 32, 
                kernel_size = c(4,4), 
                padding = "same", activation = "relu",
                input_shape = c(28, 28, 1)
  ) %>% 
  
  layer_max_pooling_2d(pool_size = c(3,3)) %>% 
  
  layer_conv_2d(filters = 32, 
                kernel_size = c(4,4), 
                padding = "same", activation = "relu",
                input_shape = c(28, 28, 1)
  ) %>% 
  
  layer_max_pooling_2d(pool_size = c(3,3)) %>% 
  
  layer_flatten() %>% 
  
  layer_dense(units = 16, 
              activation = "relu") %>% 
  
  layer_dense(units = 10, 
              activation = "softmax",
              name = "Output"
  )

model

# Compilamos el modelo:
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_adam(lr = 0.001), 
          metrics = "accuracy"
  )

train_history <- model %>% 
  fit(x = train_x_keras,y = train_y, epochs = 100, batch_size = 32, validation_split = 0.1,verbose = 1)

# Mostramos el comportamiento del model en cada epoch:
plot(train_history)


pred_test <- predict_classes(model, test_x_keras)

data.frame(
  Accuracy = accuracy_vec(truth = as.factor(data_test$C785), 
                          estimate = as.factor(pred_test)
  ),
  Recall = sens_vec(truth = as.factor(data_test$C785),
                    estimate = as.factor(pred_test)
  ),
  Precision = precision_vec(truth = as.factor(data_test$C785), 
                            estimate = as.factor(pred_test)
  ),
  F1 = f_meas_vec(truth = as.factor(data_test$C785), 
                  estimate = as.factor(pred_test)
  )
) %>% 
  mutate_all(scales::percent, accuracy = 0.01)

pred_test <- predict_classes(model, val_x_keras)
View(pred_test)

# Guardamos los datos:
write.csv(pred_test,'MNIST_CNN_PRED'
)


#### Logistic Regression (Primera Parte):####

# Cargamos los paquetes:
library(glmnet)
library(ggplot2)
library(caret)
library(doParallel)
library(IMIFA)

# Cargamos los datos:
train=read.csv("C:/Users/Claudia/Desktop/MNISTtrain_40000.csv")
test=read.csv("C:/Users/Claudia/Desktop/MNISTtest_9000.csv")
val=read.csv("C:/Users/Claudia/Desktop/MNISTvalidate_11000.csv")

# Hacemos la ultima columna de tipo factor:
y=train$C785
x=as.matrix(train[,1:784])

# Hacemos matrices a los datos de prueba:
testx=as.matrix(test[,1:784])
testy=test$C785

# Alpha =0.1:
fit=cv.glmnet(x,y,family = "multinomial",type.measure = "class",alpha=0.1 )
plot(fit)

# Hacemos predict para lambda min y 1se:
pred=predict(fit,x,s=fit$lambda.min,type="class")

# Comprobamos los primeros 6 valores: 
par(mfrow=c(2,3))
for(i in 1:6){
  show_digit(testx[i,])
  title(sprintf("prediction = %s",pred[i]))
}

# Calculamos la exactitud:
mean(testy==pred)

# Resultados de los errores globales:
tst_err1=1-0.9212222 #(lambda.1se) 
train_err1=1-0.93355 #(lambda.1se)
tst_err1=1-0.9205556 #(lambda.min) 
train_err1=1-0.939425  #(lambda.min)

# De manera análoga se trabaja para el cconjunto de test y de traain variando el
# alpha, obteniendo lo siguiente:

# Primero definimos la siguiete funcion para calculaar los erores por clase:
err_rate=function(tab){
  err=c()
  for (i in 1:10) {
    err=c(err,1-tab[i,i]/sum(tab[i,]))
  }
  return(err)
}

# Usamos la funcion table:
tab=table(y,pred)

# Al resultado de la funcion se le aplica la funcion que definimos:
err_rate(tab) 

# Guardamos y cargamos los datos de la siguiente forma:
#write.csv(pred2,'MNIST_LR_PRED')
#saveRDS(fit, "glm92.rds")
#my_model <- readRDS("glm92.rds")
#plot(my_model)

# Finalemente se obtienen estos errores por clase:
er_test_lambdamin1=c(0.01910112, 0.02944063, 0.10109890, 0.12407211, 0.06071019,
                     0.11925466, 0.05647841, 0.06601732, 0.13461538, 0.09305065)

er_test_lambda1se1=c(0.01910112, 0.02944063, 0.09890110, 0.12513256, 0.05612829,
                     0.12670807, 0.05758583, 0.06277056, 0.13009050, 0.09305065)

er_train_lambdamin1=c(0.02189967, 0.02025373, 0.08079537, 0.08758234, 0.05318333,
                      0.09415317, 0.03156019, 0.05083113, 0.09636364, 0.07678926)

er_train_lambda1se1=c(0.02470079, 0.02359225, 0.08884974, 0.09416931, 0.05957556,
                      0.10101565, 0.03385085, 0.05854011, 0.10207792, 0.08598410)


# Alpha=0.2:
fit2=cv.glmnet(x,y,family = "multinomial",type.measure = "class",alpha=0.2 )
plot(fit2)

# Hacemos predict:
pred2=predict(fit2,x,s=fit2$lambda.min,type="class")

# Mostramos los valores reales y la prediccion:
for(i in 1:6){
  show_digit(testx[i,])
  title(sprintf("prediction = %s",pred5[i]))
}

# Calculamos el accuracy:
mean(testy==pred2)

# Resultados de los errores globales:
tst_err2=1-0.9215556 #(lambda.1se) 
train_err2=1-0.932375 #(lambda.1se)
tst_err2=1-0.9208889 #(lambda.min) 
train_err2=1- 0.9377  #(lambda.min)

tab2=table(y,pred2)
err_rate(tab2)

# Al igual que en el anterior calculamos los errores por clase:
er_test_lambdamin2=c(0.02134831, 0.03042198, 0.10000000, 0.12513256, 0.05841924,
                     0.11925466, 0.05426357, 0.06168831, 0.13348416, 0.09658422)

er_test_lambda1se2=c(0.01910112, 0.02845927, 0.09890110, 0.12407211, 0.05727377,
                     0.12422360, 0.05758583, 0.06168831, 0.13009050, 0.09305065)

er_train_lambda1se2=c(0.02495544, 0.02336969, 0.08985653, 0.09490120, 0.06110969, 
                      0.10458413, 0.03385085, 0.05878102, 0.10493506, 0.08822068)

er_train_lambdamin2=c(0.02368220, 0.02136657, 0.08280896, 0.08953403, 0.05548453,
                      0.09744716, 0.03232375, 0.05227656, 0.09662338,0.07927435)

# Para guardar y usar el modelo se usaron ambas funciones:
#saveRDS(fit2, "glm92_2.rds")
my_model <- readRDS("glm92_2.rds")

# Alpha=0.3:
fit3=cv.glmnet(x,y,family = "multinomial",type.measure = "class",alpha=0.3 )
plot(fit3)

# Hacemos predict:
pred3=predict(my_model,x,s=my_model$lambda.min,type="class")

# Visualizamos los valores:
for(i in 1:6){
  show_digit(testx[i,])
  title(sprintf("prediction = %s",pred4[i]))
}

# Calculamos el accuracy:
mean(testy==pred4)

# Errores globales:
tst_err3=1-0.9212222 #(lambda.1se) 
train_err3=1-0.932975 #(lambda.1se)
tst_err3=1-0.9203333 #(lambda.min) 
train_err3=1- 0.94105  #(lambda.min)

#write.csv(pred4,'MNIST_LR_PRED2')
#saveRDS(fit2, "glm92_2.rds")
#my_model <- readRDS("glm92_2.rds")
#plot(my_model)

tab3=table(y,pred3)
err_rate(tab3)

# Al igual que en el anterior calculamos los errores por clase:
er_test_lambdamin3=c(0.02022472, 0.03238469, 0.09890110, 0.12725345, 0.05727377,
                     0.11801242, 0.05758583, 0.06385281, 0.13348416, 0.09658422)

er_test_lambda1se3=c( 0.02022472, 0.02845927, 0.10000000, 0.12407211, 0.05841924,
                      0.12298137, 0.05647841, 0.06060606, 0.13122172,0.09540636)

er_train_lambda1se3=c(0.02495544, 0.02270198, 0.08809464, 0.09465723, 0.06110969,
                      0.10403514, 0.03359633, 0.05854011, 0.10285714, 0.08797217)

er_train_lambdamin3=c(0.02062643, 0.02092143, 0.07827838, 0.08660649, 0.05190488,
                      0.09223168, 0.02926953, 0.04986750, 0.09350649, 0.07355865)


# Alpha=0.4:
fit4=cv.glmnet(x,y,family = "multinomial",type.measure = "class",alpha=0.4 )
plot(fit4)

#saveRDS(fit4, "glm92_4.rds")
#my_model <- readRDS("glm92_4.rds")

# Hacemos predict:
pred4=predict(fit4,x,s=fit4$lambda.1se,type="class")

# Calculamos el accuracy:
mean(testy==pred4)

# Errores globales:
tst_err4=1-0.9208889 #(lambda.1se) 
train_err4=1-0.93255 #(lambda.1se)
tst_err4=1-0.9208889 #(lambda.min) 
train_err4=1-0.937125  #(lambda.min)

# Tabla de contingencia:
tab4=table(y,pred4)
err_rate(tab4)

# Finalmente se obtuvieron los siguientes errores por clase:
er_test_lambdamin4=c(0.02134831, 0.03042198, 0.10109890, 0.12513256, 0.05841924,
                     0.11801242, 0.05426357, 0.06168831, 0.13348416, 0.09658422)

er_test_lambda1se4=c(0.02022472, 0.02747792, 0.10000000, 0.12619300, 0.05841924,
                     0.12173913, 0.05869324, 0.06060606, 0.13122172, 0.09658422)

er_train_lambda1se4=c(0.02521008, 0.02270198, 0.08859804, 0.09538912, 0.06110969,
                      0.10458413, 0.03410537, 0.05878102, 0.10441558, 0.08797217)

er_train_lambdamin4=c(0.02215432, 0.02181171, 0.08306066, 0.09099780, 0.05625160,
                      0.10019215, 0.03283278, 0.05251747, 0.09636364, 0.08051690)


# Empleamos el mejor modelo (Alpha =0.2) para hacer el predict sobre la base de 
# datos de validacion:
pre_val=predict(my_model,valx,s=my_model$lambda.1se,type="class")
head(pre_val)

# La guardamos en formato de csv:
write.csv(pre_val,"val_pred_GLM")


library(glmnet)
fit2 <- glmnet(x=as.matrix(predictors2), y=response2,alpha=0.6, family = "multinomial")
cverror <- cv.glmnet(
  x=as.matrix(predictors2) ,
  y =response2 ,
  alpha  = 0.6,
  nfolds = 6,
  type.measure = "class",
  family = "multinomial"
)

pred.glm= assess.glmnet(fit2,s=cverror$lambda.min ,newx = as.matrix(predictors2) , newy = response2)



predicciones= predict(fit2,s=cverror$lambda.min ,newx = as.matrix(predictors2),  type=c("class"))
table(response2,predicciones)

sum(diag(table(response2,predicciones)))



fit3 <- glmnet(x=as.matrix(predictors2), y=response2,alpha=0.7, family = "multinomial")

cverror2 <- cv.glmnet(
  +   x=as.matrix(predictors) ,
  +   y =response ,
  +   alpha  = 0.7,
  +   nfolds = 6,
  +   type.measure = "class",
  +   family = "multinomial" 
)

fit4 <- glmnet(x=as.matrix(predictors2), y=response2,alpha=0.8, family = "multinomial")

cverror2 <- cv.glmnet(
  +   x=as.matrix(predictors) ,
  +   y =response ,
  +   alpha  = 0.8,
  +   nfolds = 6,
  +   type.measure = "class",
  +   family = "multinomial" )

predicciones2= predict(fit2,s=cverror2$lambda.min ,newx = as.matrix(predictors),  type=c("class"))
table(response,predicciones2)
sum(diag(table(response,predicciones2)))/9000

predicciones3= predict(fit2,s=cverror2$lambda.min ,newx = as.matrix(predictors),  type=c("class"))
table(response,predicciones3)
sum(diag(table(response,predicciones2)))/9000

predicciones4= predict(fit2,s=cverror2$lambda.min ,newx = as.matrix(predictors),  type=c("class"))
table(response,predicciones4)
sum(diag(table(response,predicciones2)))/9000

valida_logistic= predict(fit2,s=cverror2$lambda.min ,newx = as.matrix(MNISTvalidate),  type=c("class"))
write.csv(valida_logistic, file="predicciones Logistic.csv", row.names=F) 




#La siguiente es una prueba en H2O sin embargo el tiempo de maquina que requiere es demasiado a comparacion de el paquete random forest
# por lo que este no se usara mas.
#library(h2o)
#h2o.init()
#trainh2o <- h2o.importFile("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/proyecto final/MNISTtrain_40000.csv")
#MNISTtrainh2o<- as.factor(trainh2o)

#predictors <-as.matrix( MNISTtrainh2o)
#response <- as.vector(trainh2o[,785])

#set.seed(1235)

#gluc_gbm <- h2o.gbm(x = predictors[,1:784],
# y = response,
# nfolds = 8,
#seed = 1231,
#training_frame = MNISTtrainh2o)


#--------------------------------Random forest

library(readr)
#Se leen las bases de datos

MNISTtrain <- read_csv("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/proyecto final/MNISTtrain_40000.csv")
MNISTtest <- read_csv("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/proyecto final/MNISTtest_9000.csv")
MNISTvalidate <- read_csv("C:/Users/Family/Desktop/Mat Niye/estadistica/aprendizaje estadistico/proyecto final/MNISTvalidate_11000.csv")

predictors <-MNISTtrain[,-785]
response <- as.factor(MNISTtrain$C785)

summary(predictors)


mnisistcom= rbind(MNISTtrain, MNISTtest)

###### Random Forest:#####
library(randomForest)
#-----pruebas previas-------
rf <- randomForest(x = predictors, y = response, ntree = 50)
rf100 <- randomForest(x = predictors, y = response, ntree = 100)
rf500 <- randomForest(x = predictors, y = response, ntree = 500)
rftotal <- randomForest(x = predictors, y = response,xtest= MNISTtest[,-785], ytest=as.factor(MNISTtest$C785), ntree = 2000)

?randomForest
a=predict(rf,newdata = MNISTtest[,-785])

#Entrenando con la base completa

predictors2 <-mnisistcom[,-785]
response2 <- as.factor(mnisistcom$C785)

set.seed(3632)
rf3_train <- randomForest(x = predictors, y = response, ntree = 1000,strata=MNISTtrain$C785)#Modelo que se usa
rf3 <- randomForest(x = predictors2, y = response2, ntree = 1000,strata=mnisistcom$C785)

#con un numero de variables diferentes a las anteriores 
rf4 <- randomForest(x = predictors2, y = response2, ntree = 700, strata=mnisistcom$C785,mtry=50 )
rf5 <- randomForest(x = predictors2, y = response2, ntree = 700, strata=mnisistcom$C785,mtry=100 )
rf6 <- randomForest(x = predictors2, y = response2, ntree = 700, strata=mnisistcom$C785,mtry=20 ) #NOTA: los errores estan mas grandes no usar.


#Calculo de errores
error_test=table(as.factor(MNISTtest$C785),predict(rf3_train,newdata = MNISTtest[,-785]))
confusionMatrix(predict(rf3_train,newdata = MNISTtest[,-785]), as.factor(MNISTtest$C785), dnn = c("Prediction", "Reference"))
rf.cv <- rf.crossValidation(rf3_train, predictors, response, p=1/3, n=3, ntree=1000)

#Guardar los valore para la validacion

valida_rf=predict(rf3_train,newdata = MNISTvalidate)

write.csv(valida_rf, file="predicciones Ranforest.csv", row.names=F)





