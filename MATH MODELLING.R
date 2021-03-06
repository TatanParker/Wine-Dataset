# ##1. INICIO

# Integraci�n de librer�as

rm(list=ls()) ###Limpiamos el entorno global

# Instalaci�n de paquetes y declaraci�n de librer�as

packages = c("tidyverse", "RCurl", "psych", "stats", 
             "randomForest", "glmnet", "caret","kernlab", 
             "rpart", "rpart.plot", "neuralnet", "C50",
             "doParallel", "AUC", "ggfortify", "rmdformats", 
             "corrgram", "ggplot2", "naniar", "e1071",
             "lattice", "caret", "knitr", "corrplot", 
             "kknn", "randomForest", "kernlab", "car", "xlsx", 
             "data.table", "GGally", "gplots", "kableExtra")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
invisible(lapply(packages, require, character.only = TRUE))

# Creacion de una funcion que elimina outliers en funci�n de los valores que se encuentran dentro de los percentiles altos

remove_outliers <- function(x,quant){
  
  require("dplyr")
  
  for(i in 1:11){
    
    x = mutate(x,outliers=ifelse(x[[i]] < quantile(x[[i]],quant),0,1))
    x = filter(x, outliers==0) 
    x = select(x, -outliers)
    
  }
  
  return(x)
  
}

eval = function(pred, true, plot = F, title = "") {
  rmse = sqrt(mean((pred - true)^2))
  mae = mean(abs(pred - true))
  cor = cor(pred, true)
  if (plot == TRUE) {
    par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
    diff = pred - true
    plot(jitter(true, factor = 1),
         jitter(pred, factor = 0.5),
         pch = 3, asp = 1,
         xlab = "Real", ylab = "Prediccion")
    abline(0,1, lty = 2)
    hist(diff, breaks = 20, main = NULL)
    mtext(paste0(title, "Predicci�n vs Real"), outer = TRUE)
    par(mfrow = c(1,1))}
  return(list(RMSE = rmse,
              MAE = mae,
              CORR = cor))
}

# #### Carga de los datos:

setwd("D:/MAIN/MASTER/M11/WINE")


wine = data.frame(read.xlsx("SOURCE/SOURCE.xlsx", sheetIndex = 1))

  NewNames = c("clase","acidez_fija","acidez_volatil","acidez_citrica","azucar_res","cloruros","sulfitos_libres","sulfitos_totales","densidad","ph","sulfatos","alcohol","calidad")
  
  names(wine)=NewNames

# #### Limpieza y preparaci�n de datos

# Chequeamos los NA del dataset
  
vis_miss(wine)
  
    ###Ninguno

head(wine)

summary(wine)

# Se observan valores relativamente esperables, seg�n los valores comprendidos como "usuales" seg�n el estudio preliminar que hemos realizado. Valores importantes como la densidad, alcohol, pH, sulfitos totales, cloruros o acidez se encuentran dentro del rango de valores esperado.
# Tambi�n puede observarse que hay m�ximos que muestran valores fuera de los percentiles altos y que se candidatan como **outliers**, como es el caso de los sulfitos libres, totales, y el azucar residual.
# En general, el set de datos est� bastante limpio y apenas requiere de limpieza, aparte de los outliers, que no se eliminar�n aun hasta hacer el an�lisis exploratorio, a excepci�n de los m�s fuertes. Se proceder� a la divisi�n por tipo de vino.

str(wine)
  
  #Se confirman que s�lo el atributo clase, agregado al dataset original como factor



# Se crean pues, los datasets de las dos clases de vino por separado, eliminando el atributo clase y eliminando outliers fuertes:

unloadNamespace("seriation")

white = wine %>% filter(clase=="WHITE") %>% select(-clase)

  white = remove_outliers(white, 0.999) 
  
  summary(white)
  
  ##Eliminamos los outliers m�s fuertes (por encima del percentil 99.9)

red = wine %>% filter(clase=="RED") %>% select(-clase)

  red = remove_outliers(red, 0.9999) 

# ##2. AN�LISIS DE VINO BLANCO (PREDICCI�N)

# ### 0. Preparaci�n de datos para modelaje

# Recarga del dataset y normalizaci�n

set.seed(1) 

normalize = function(x){(x-min(x))/(max(x)-min(x))} # Funci�n para normalizar datasets

# ##### Creaci�n y normalizaci�n de train y test para predicci�n (Vino Blanco)

Index <- sample(1:nrow(white), 0.8*nrow(white))  
whitetrain <- white[Index, ]  
whitetest  <- white[-Index, ] 

multi.hist(whitetrain)

whitetrainN = data.frame(apply(whitetrain[,-12],2, normalize),
                         calidad = whitetrain[,12])

whitetrain.min = apply(whitetrain[,-12], 2, min)
whitetrain.max = apply(whitetrain[,-12], 2, max)
whitetestN = data.frame(sweep(whitetest, 2, c(whitetrain.min, 0)) %>% 
                         sweep(2, c(whitetrain.max-whitetrain.min, 1), FUN = "/"))

summary(whitetrainN)
summary(whitetestN)

# Test Shapiro, para testear la normalidad de nuestra distribuci�n

shapiro.test(white$calidad)

# Se rechaza la hip�tesis nula, ya que p es muy inferior a 0,05. Esto quiere decir claramente que uno (o m�s de un) de los predictores est� altamente relacionado con la respuesta


# ### Modelos de Regresi�n

# ### 1. Regresi�n lineal

# #### 1.1. Regresi�n lineal simple

linealwhite <- lm(calidad ~ . , whitetrainN) 
summary(linealwhite)

# La R2 del modelo es de 0.2874, de forma que este modelo no se ajusta especialmente a la realidad, ni puede contemplar gran parte de la dispersi�n real de los atributos.

# A continuaci�n se hace un an�lisis de varianza a fin de encontrar outliers que puedan eliminarse para mejorarlo

par(mfrow=c(2,3))
lapply(1:6, function(x) plot(linealwhite, which=x, 
                             labels.id= 1:nrow(whitetrainN))) %>% invisible()


# Los gr�ficos Residuals vs Fitted muestran si los residuos tienen patrones no lineales. Los residuos alrededor de una l�nea horizontal sin patrones distintos, es una buena indicaci�n de que no tenemos relaciones no lineales.
# 
# La gr�fica QQ normal muestra los residuos que se ajustan a la l�nea. Por lo tanto, puede llamarlo residuos distribuidos normalmente.
# 
# El gr�fico de Scale-location muestra si los residuos se distribuyen por igual a lo largo de los rangos de los predictores. As� es como puede verificar el supuesto de varianza igual (homocedasticidad). Es bueno si ve una l�nea horizontal con puntos de dispersi�n iguales (al azar).
# 
# La trama Residuals vs Leverage tiene un aspecto t�pico cuando hay alg�n caso influyente. Apenas puede ver las l�neas de distancia de Cook (una l�nea discontinua roja) porque todos los casos est�n dentro de la distancia de Cook.
# 
# El gr�fico de Cook's distance resalta las observaciones m�s at�picas del modelo, se contemplan 3 observaciones que se encuentran visiblemente dispersas del resto del conjunto, as� que los eliminamos


whitetrainN=whitetrainN[-c(257,1681,3277),]

linealwhite <- lm(calidad ~ . , whitetrainN) 
summary(linealwhite)

par(mfrow=c(2,3))
lapply(1:6, function(x) plot(linealtrain, which=x, 
                             labels.id= 1:nrow(whitetrainN))) %>% invisible()


# La R2 no ha variado, de manera que se deja el modelo como est�

linealwhitePredictor = predict(linealwhite, whitetestN[,-12])
linealwhiteEvaluator = eval(linealwhitePredictor,
                           whitetestN[,12], plot = T, title = "lm: ")


unlist(linealwhiteEvaluator)

# El RMSE o error cuadr�tico medio es de 0,7, y trat�ndose de un dataset con m�s de 1000 unidades, no se trata de un valor especialmente bajo.
# El MAE es de 0,57, que en la escala de 3-9 que nos encontramos, es algo mayor al 10% del orden de magnitud, lo cual no es tampoco un error fatal, pero es inseguro.
# La correlaci�n es de 0,52, que es aceptable si se trata de evaluar la dependencia de ambas, pero no suficiente para considerarlo �ptimo el modelo

# El histograma sirve para mostrar que este modelo es bueno en las regiones centrales, donde es l�gico, entre otras cosas, porque se dispone de mayor cantidad de datos, pero tambi�n por la baja correlaci�n entre mucho de sus predictores. El modelo, por tanto ser�a relativamente aceptable para predecir calidad de vinos medios.

# Para solucionar el problema de este modelo, se puede hacer lo siguiente:

# - Estudiar la colinealidad y asimetr�a del dataset, y mejorar el modelo eliminando predictores de alta colinealidad
# 
# - Probar modelos polinomiales, regresi�n cuadr�tica
# 
# - Probar a categorizar variables


# #### 1.2 - Regresi�n lineal mejorada:

# Se estudia la multicolinealidad, y se mejora el modelo

# En multicolinealidad (colinealidad entre tres o m�s variables) implica que hay redundancia entre las variables predictoras, y el modelo se vuelve inestable.

# La multicolinealidad se va a evaluar calculando un puntaje llamado factor de inflaci�n de la varianza (o VIF), que mide cu�nto se infla la varianza de un coeficiente de regresi�n debido a la multicolinealidad en el modelo.
# El VIF de un predictor es una medida de la inflaci�n de la varianza que se predice a partir de una regresi�n lineal utilizando los otros predictores.

vif(linealtrain)

# La densidad es el predictor con mayor colinealidad, ya que conocemos la relacion de dependencia que tiene con el alcohol, as� que va a retirarse
# 

linealwhite1 <- lm(calidad ~ . -acidez_citrica -densidad, whitetrainN) 
summary(linealwhite1)

vif(linealwhite1) # Ahora ha mejorado mucho el an�lisis de varianza, se procede a visualizarlo

par(mfrow=c(2,3))
lapply(1:6, function(x) plot(linealwhite1, which=x, 
                             labels.id= 1:nrow(whitetrainN))) %>% invisible()

linealwhitePredictor1 = predict(linealwhite1, whitetestN[,-12])
linealwhiteEvaluator1 = eval(linealwhitePredictor1,
                            whitetestN[,12], plot = T, title = "lm: ")

unlist(linealwhiteEvaluator)
unlist(linealwhiteEvaluator1)

# Se ha mejorado la inflaci�n de la varianza sin obtener cambios en los errores del modelo.
#####

set.seed(143)

PoissonWhite = glm(calidad~., data = whitetrainN,
                    family = "poisson")

summary(PoissonWhite)

par(mfrow=c(2,3))
lapply(1:6, function(x) plot(PoissonWhite, which=x, 
                             labels.id= 1:nrow(whitetrainN))) %>% invisible()

PoissonWhitePredictor = predict(PoissonWhite,whitetestN, type = "response")
PoissonWhiteEvaluator = eval(PoissonWhitePredictor,
                             whitetest[,12], plot = T, title = "Poisson: ")

unlist(PoissonWhiteEvaluator)

######

# ### 2. Regresi�n M�ltiple (cuadr�tica-binomial)

quadraticWhite = lm(calidad~ poly(acidez_fija, 2) + 
             poly(acidez_volatil,2) +
             poly(acidez_citrica,2) +
             poly(cloruros,2) + 
             poly(sulfitos_libres,2) +
             poly(sulfitos_totales,2) + 
              
             poly(azucar_res,2) +  
             poly(densidad,2) + 
             poly(ph,2) + 
             poly(sulfatos,2) + 
             poly(alcohol,2), 
           data = whitetrainN)

summary(quadraticWhite)


quadraticWhitePredictor = predict(quadraticWhite, whitetestN[,-12])
quadraticWhiteEvaluator = eval(quadraticWhitePredictor,
                             whitetestN$calidad, plot = T, title = "qm: ")

unlist(quadraticWhiteEvaluator)

# El modelo mejora, pero no es suficiente

# ###3. Modelo ANOVA, categorizando las variables

# Se categorizan los predictores con las siguientes reglas:
# 
# 0 - 10%
# 
# 10 - 30%
#   
# 30 - 50%
# 
# 50% - 75%
#   
# 90 - 100%

c1 = apply(whitetrainN, 2, function(x) quantile(x, 0.10))
c2 = apply(whitetrainN, 2, function(x) quantile(x, 0.30))
c3 = apply(whitetrainN, 2, function(x) quantile(x, 0.50))
c4 = apply(whitetrainN, 2, function(x) quantile(x, 0.75))
c5 = apply(whitetrainN, 2, function(x) quantile(x, 0.9))

categorize = function(dataset = whitetrainN) {
  df.cat = dataset
  for (i in 1:(ncol(dataset)-1)){
    col = dataset[,i]
    cat = case_when(col<c1[i]                    ~ "0",
                    col>=c1[i] & col<c2[i]       ~ "1",
                    col>=c2[i] & col<c3[i]       ~ "2",
                    col>=c3[i] & col<c4[i]       ~ "3",
                    col>=c4[i] & col<c5[i]       ~ "4",
                    col>=c5[i]                   ~ "5")
    df.cat[,i] = cat
  }
  return(df.cat)
}

whitetrainCat = categorize(whitetrainN)
whitetestCat = categorize(whitetestN)

summary(whitetrainCat)

head(whitetrainCat)

anovaWhite = lm(calidad ~ ., data = whitetrainCat)

summary(anovaWhite)

anovaWhitePredictor = predict(anovaWhite, whitetestCat[,-12])
anovaWhiteEvaluator = eval(anovaWhitePredictor,
                               whitetestCat$calidad, plot = T, title = "cm: ")

unlist(anovaWhiteEvaluator)

# Se puede comprobar que los resultados son similares a los conseguidos en el modelo binomial

# ###4. Interacci�n y selecci�n de variables (paso a paso)

InteractWhite = lm(calidad~ .^2, data = whitetrain)
summary(InteractWhite)

## Se efect�a el m�todo paso a paso

StepWhite = lm(calidad ~ 1, data = whitetrainN)
InteractWhiteStep = step(StepWhite, ~ (acidez_fija + acidez_volatil + 
                                      acidez_citrica + azucar_res +  cloruros + sulfitos_libres +
                                      sulfitos_totales + densidad + ph + sulfatos + alcohol)^2, 
                            direction = "both", trace = 0)
summary(InteractWhiteStep)

InteractWhiteStepPredictor = predict(InteractWhiteStep, whitetestN[,-12])
InteractWhiteStepEvaluator = eval(InteractWhiteStepPredictor, whitetestN$calidad, plot=T, title="sm: ")

unlist(InteractWhiteStepEvaluator)


# Con este modelo se ha mejorado notablemente la correlaci�n, el error tambi�n ha disminuido

# ### Modelos de Clasificaci�n

# ### 5. Random Forest

# #### 5.1. Random Forest con libreria RandomForest

# Se va a construir un modelo de �rboles de decisi�n, agrupando 1000 �rboles de decisi�n, cada �rbol con 3 variables.

RandForestWhite = randomForest(calidad~., data = whitetrainN, ntree = 1000, mtry = sqrt(12))

RandForestWhite

RandForestWhitePredictor = predict(RandForestWhite, whitetestN[,-12])
RandForestWhiteEvaluator = eval(RandForestWhitePredictor, whitetestN$calidad, plot = T, title = "rfm: ")

RandForestWhiteEvaluator

# Con este cambio de modelo, y tal como se puede observar, los errores han descendido en torno al 8-10% y la correlaci�n ha subido en torno a .15 puntos

# #### 5.2. Random Forest con librer�a Caret

# Se usar� la funci�n train del package "caret" para aplicar una validaci�n cruzada con las siguientes caracter�sticas:
# Se ahora en adelante, los modelos creados con la libreria caret se realizar�n con el dataset sin normalizar, ya que usaremos la funci�n preproceso donde centraremos y escalaremos previamente el dataset
# *	10 veces con 4 repeticiones
# *	Se incluyen 2, 4-6 variables respectivamente por nivel de �rbol.
# A fin de que se optimicen los par�metros del modelo. Se evaluar� posteriormente con el RMSE. No se aplicar�n muchas combinaciones por cuestiones de memoria. Se usar� el paquete doParallel para mejorar el rendimiento.

# Se usar� la funcion trainControl para ajustar los hiperpar�metros. Tras diferentes entrenamientos, se escogen estos valores porque se encuentran dentro de los m�rgenes de �xito, sin que provoque un consumo computacional grande

controlRandForest = trainControl(method = "repeatedcv", number = 10, repeats = 3)

matrizRandForest = expand.grid(.mtry = c(2, 3, 6))

set.seed(1)

clusterRandForest = makePSOCKcluster(4)
registerDoParallel(clusterRandForest)

RandForestCrossValWhite = train(calidad~., data = whitetrain,
                method = 'rf',
                metric = "RMSE",
                trControl = controlRandForest,
                tuneGrid = matrizRandForest,
                preProcess = c("center", "scale"))

stopCluster(clusterRandForest)

plot(RandForestCrossValWhite)

RandForestCrossValWhite$bestTune

RandForestCrossValWhitePredictor = predict(RandForestCrossValWhite, whitetest[,-12])
RandForestCrossValWhiteEvaluator = eval(RandForestCrossValWhitePredictor, whitetest[,12], plot = T, title = "rfcvm: ")

unlist(RandForestCrossValWhiteEvaluator)

# ### 6. K-Nearest Neighbours

# #### 6.1. K-Nearest Neighbours con librer�a kknn

require(class)

set.seed(12)

KKNNWhite <- train.kknn(calidad~., 
                        whitetrainN, 
                        ks = c(3, 5, 7 ,9, 11, 17),
                        distance = c(1, 2), 
                        kernel =c("rectangular", "gaussian", "cos"))

summary(KKNNWhite)

KKNNWhitePredictor = predict(KKNNWhite, whitetestN[,-12])
KKNNWhiteEvaluator = eval(KKNNWhitePredictor, whitetestN[,12], plot = T, title = "KKNN: ")

unlist(KKNNWhiteEvaluator)


# #### 6.2. K-Nearest Neighbors con librer�a Caret

# Este modelo tambi�n es sugerido por el estudio considerado como fuente del proyecto. Este es un modelo de clasificaci�n supervisada no param�trico, de modo que es un modelo perfecto a entrenar para un set de datos de variables continuas como el nuestro.

# Para KKNN, se utilizar�n 5 km�x, 2 distancias y 3 valores del n�cleo. Para el valor de la distancia, 1 es la distancia de Manhattan y 2 es la distancia euclidiana (m�s favorable).

set.seed(1)

clusterKKNN = makePSOCKcluster(4)

registerDoParallel(clusterKKNN)

controlKKNN = trainControl(method = "repeatedcv", repeats = 5, classProbs = TRUE)

matrizKKNN = expand.grid(kmax = c(3, 5, 7 ,9, 11, 17), distance = c(1, 2),
                         kernel = c("rectangular", "gaussian", "cos"))

KKNNCrossValWhite <- train(calidad ~ ., data = whitetrain, 
                    method = "kknn",
                    trControl = controlKKNN, 
                    tuneGrid = matrizKKNN,
                    metric = "RMSE",
                    preProcess = c("center", "scale"))

stopCluster(clusterKKNN)

plot(KKNNCrossValWhite)

KKNNCrossValWhite$bestTune

KKNNCrossValWhitePredictor = predict(KKNNCrossValWhite, whitetest[,-12])
KKNNCrossValWhiteEvaluator = eval(KKNNCrossValWhitePredictor, whitetest[,12], plot = T, title = "KKNNcvm: ")

unlist(KKNNCrossValWhiteEvaluator)


# ### 7. Support Vector Machine

# Este modelo es uno de los modelos principales considerados en el estudio (Modeling wine preferences by data mining), citado en la introducci�n y fuente principal del presente dataset:
# Las SVM presentan ventajas te�ricas sobre NN (vecinos cercanos), como la ausencia de m�nimos locales en la fase de aprendizaje. En efecto, el SVM fue considerado recientemente uno de los algoritmos de Data mining esenciales. Si bien el modelo MR es m�s f�cil de interpretar, a�n es posible extraer m�s conocimiento de NN y SVM, dado en t�rminos de importancia variable de entrada

# Se usar� el model ksvm de la libreria kernel, con la funci�hn kernel centrada en base radial gaussiana, que es la que mejor se ajusta al modelo que se quiere.


SVMWhite=ksvm(calidad ~ ., 
              data = whitetrainN, 
              scaled = F,
              kernel = "rbfdot", 
              C = 1)

SVMWhitePredictor = predict(SVMWhite, whitetestN[,-12])
SVMWhiteEvaluator = eval(SVMWhitePredictor, whitetestN[,12], plot = T, title = "SVM: ")

unlist(SVMWhiteEvaluator)

# Este modelo, y contra pron�stico, en relaci�n a las suposiciones extra�das del estudio, no parece ser m�s adecuado que el de �rbol aleatorio. Se usar� la metolog�a de CrossValidation y optimizaci�n de hiperpar�metros para mejorar el obtenido hasta ahora.

# Se utilizar� la funci�n de base Radial como n�cleo del modelo SVM 

# Se har� exactamente lo mismo pero usando el m�todo svmRadial" de la libreria caret

set.seed(1)

clusterSVM = makePSOCKcluster(4)

registerDoParallel(clusterSVM)

controlSVM = trainControl(method = "repeatedcv", number = 5, repeats = 5)

# Para la matriz del modelo voy a utilizar los siguientes hiperpar�metros:

# *Sigma = El ancho del inverso del n�cleo, para suavizar. Tambi�n conocida como distribuci�n acumulativa inversa normal.
# *C = Coste de desclasificaci�n

matrizSVM = expand.grid(C = 2^(1:3), sigma = seq(0.25, 2, length = 8))

# Ambos par�metros han sido ajustados ya que en los valores de sigma exist�a un pico de precisi�n entre los valores escogidos y el valor de C, que m�s all� del intervalo escogido no parec�an revelar m�s datos significativos en el entrenamiento. 

SVMCrossValWhite = train(calidad~., data = whitetrainN,
                                method = 'svmRadial',
                                trControl = controlSVM,
                                tuneGrid = matrizSVM)


stopCluster(clusterSVM)

plot(SVMCrossValWhite)

SVMCrossValWhite$bestTune

# Se comprueba que los mejores valores de entrenamiento corresponden a sigma = 0,75 y C = 2. En el se encuentran los valores de RMSE m�s bajo, tal como se observa en el m�nimo relativo.

SVMCrossValWhitePredictor = predict(SVMCrossValWhite, whitetestN[,-12])
SVMCrossValWhiteEvaluator = eval(SVMCrossValWhitePredictor, whitetestN[,12], plot = T, title = "SVMcvm: ")

unlist(SVMCrossValWhiteEvaluator)

# ### 8. Regression Tree

# Se estudia, como �ltimo modelo a usar, el �rbol de regresi�n con la librer�a rpart.

RTWhite = rpart(calidad~., data = whitetrainN)
rpart.plot(RTWhite)

RTWhitePredictor = predict(RTWhite, whitetestN[,-12])
RTWhiteEvaluator = eval(RTWhitePredictor, whitetestN[,12], plot = T, title = "RT: ")

unlist(RTWhiteEvaluator)

# ### 9. Sumario

   cbind(Lineal = unlist(linealwhiteEvaluator1),
         Multiple = unlist(quadraticWhiteEvaluator),
         Poisson = unlist(PoissonWhiteEvaluator),
         Anova = unlist(anovaWhiteEvaluator),
         Step = unlist(InteractWhiteStepEvaluator),
         RandForest = unlist(RandForestWhiteEvaluator),
         RandForestCV = unlist(RandForestCrossValWhiteEvaluator),
         KKNN = unlist(KKNNWhiteEvaluator),
         KKNNCV = unlist(KKNNCrossValWhiteEvaluator),
         SVM = unlist(SVMWhiteEvaluator),
         SVMCV = unlist(SVMCrossValWhiteEvaluator),
         RT = unlist(RTWhiteEvaluator) )%>% round(3) %>% kable() %>% kable_styling()

# El modelo de bosque aleatorio arroja sin lugar a dudas los mejores valores respecto a las variables usadas para su evaluaci�n, principalmente en lo relativo al error absoluto de la media (MAE), que ha conseguido minimizarse con mayor distancia del resto de modelos.
   
# En los histogramas generados por la funci�n eval() dejan de manifiesto que no hay ning�n modelo que pueda satisfacer aceptablemente una predicci�n o clasificaci�n en zonas de respuesta perif�ricas (calidad bajas o altas)
   
# ##2. AN�LISIS DE VINO TINTO (CLASIFICACI�N)

# ### 0. Preparaci�n de datos para modelaje

# ##### Creaci�n de train y test para clasificaci�n (Vino Tinto)

wineC = wine
whiteC = wineC %>% filter(clase=="WHITE") %>% select(-clase)
redC = wineC %>% filter(clase=="RED") %>% select(-clase)
redC$calidad = as.factor(redC$calidad)

summary(redC)

IndexC = createDataPartition(redC$calidad, p = 0.8, list = F)
redtrain = redC[IndexC,]
redtest = redC[-IndexC,]

# Para simplificar, y no sobrecargar el proyecto de datos, para el an�lisis del vino tinto solo se usar�n los 3 modelos de clasificaci�n con mayor �xito creados para el an�lisis de vino blanco: K-Nearest Neighbors, RandomForest y SVM con la librer�a Caret

# ### 1. Random Forest

set.seed(1)

controlRandForest = trainControl(method = "repeatedcv", number = 10, repeats = 3)

matrizRandForest = expand.grid(.mtry = c(1:11)) ## Cambiaremos un poco respecto al calculo con el vino blanco

clusterRandForest = makePSOCKcluster(4)
registerDoParallel(clusterRandForest)

RandForestCrossValRed = train(calidad~., data = redtrain,
                                method = 'rf',
                                metric = "Accuracy",
                                trControl = controlRandForest,
                                tuneGrid = matrizRandForest,
                                preProcess = c("center", "scale"))

stopCluster(clusterRandForest)

plot(RandForestCrossValRed)

RandForestCrossValRed$bestTune

RandForestCrossValRedPredictor = predict(RandForestCrossValRed, redtest)
RandForestCrossValRedConfMatrix = confusionMatrix(RandForestCrossValRedPredictor, redtest$calidad)
RandForestCrossValRedConfMatrix

RandForestCrossValRedCF = RandForestCrossValRedConfMatrix$table %>% melt()
ggplot(RandForestCrossValRedCF, aes(Reference, y = Prediction))+
  geom_tile(aes(fill = log(value)), color="black")+
  geom_text(aes(label = value)) + 
  scale_fill_gradient(low = "lightblue", high = "steelblue") + 
  labs(title = "Matriz de confusi�n RandForest")

# ### 3. K-Nearest Neighbours

# Se aprovechan las matrices y configuraciones de control usadas anteriormente con el vino blanco

set.seed(1)

clusterKKNN = makePSOCKcluster(4)

registerDoParallel(clusterKKNN)

controlKKNN = trainControl(method = "repeatedcv", repeats = 5, number = 7)

matrizKKNN = expand.grid(kmax = c(3, 5, 7 ,9, 11, 17), distance = c(1, 2),
                         kernel = c("rectangular", "gaussian", "cos"))

KKNNCrossValRed <- train(calidad ~ ., data = redtrain, 
                         method = "kknn",
                         trControl = controlKKNN, 
                         tuneGrid = matrizKKNN,
                         metric = "Accuracy",
                         preProcess = c("center", "scale"))

stopCluster(clusterKKNN)

summary(KKNNCrossValRed)

plot(KKNNCrossValRed)

KKNNCrossValRed$bestTune

KKNNCrossValRedPredictor <- predict(KKNNCrossValRed, redtest)
KKNNCrossValRedConfMatrix = confusionMatrix(KKNNCrossValRedPredictor, redtest$calidad)
KKNNCrossValRedConfMatrix$overall

KKNNCrossValRedCF = KKNNCrossValRedConfMatrix$table %>% melt()
ggplot(KKNNCrossValRedCF, aes(Reference, y = Prediction))+
  geom_tile(aes(fill = log(value)), color="black")+
  geom_text(aes(label = value)) + 
  scale_fill_gradient(low = "lightblue", high = "steelblue") + 

  labs(title = "Matriz de confusi�n KKNN")

# ### 4. Support Vector Machine

# Misma filosof�a que usada con el Vino Blanco

set.seed(1)

clusterSVM = makePSOCKcluster(4)

registerDoParallel(clusterSVM)

controlSVM = trainControl(method = "repeatedcv", number = 5, repeats = 5)

matrizSVM = expand.grid(C = 2^(1:3), sigma = seq(0.25, 2, length = 10))

# Ambos par�metros han sido ajustados ya que en los valores de sigma exist�a un pico de precisi�n entre los valores escogidos y el valor de C, que m�s all� del intervalo escogido no parec�an revelar m�s datos significativos en el entrenamiento. 

SVMCrossValRed = train(calidad~., data = redtrain,
                         method = 'svmRadial',
                         trControl = controlSVM,
                         tuneGrid = matrizSVM)

stopCluster(clusterSVM)

plot(SVMCrossValRed)

SVMCrossValRed$bestTune

SVMCrossValRedPredictor = predict(SVMCrossValRed, redtest)
SVMCrossValRedConfMatrix = confusionMatrix(SVMCrossValRedPredictor, redtest$calidad)
SVMCrossValRedConfMatrix

SVMCrossValRedCF = SVMCrossValRedConfMatrix$table %>% melt()
ggplot(SVMCrossValRedCF, aes(Reference, y = Prediction))+
  geom_tile(aes(fill = log(value)), color="black")+
  geom_text(aes(label = value)) + 
  scale_fill_gradient(low = "lightblue", high = "steelblue") + 
  
  labs(title = "Matriz de confusi�n SVM")

# ### 5. Sumario

rbind(RandForest = RandForestCrossValRedConfMatrix$overall %>% round(3), 
      KKNN = KKNNCrossValRedConfMatrix$overall %>% round(3), 
      SVM = SVMCrossValRedConfMatrix$overall%>% round(3))

# El modelo que mejor se ajusta a la realidad es, de nuevo, el modelo Random Forest, con una precisi�n del 69,7% y un Kappa por encima del 50% (que es una medida la precisi�n del sistema respecto a la precisi�n de un sistema aleatorio). 
# Sin embargo, si observamos en la matriz de confusi�n completa de los modelos, y tal y como se hab�a predicho en el an�lisis exploratorio, y en el an�lisis del vino blanco, ninguno de los modelos tiene suficiente sensibilidad para clasificar los vinos de calidades bajas o altas (perif�ricas).


# ##3. ANALISIS FINAL

# #### RandomForest en Clasificaci�n

# Como ejercicio final del proyecto, se crear� un modelo RandomForest de clasificaci�n simplicado, orientado a un dataset de respuesta binaria (calidad >=7) que se crear� a partir del original, para ambos tipos de vino y finalmente se evaluar� con la matriz de confusi�n

wineF = wine %>% 
  mutate(excelencia=ifelse(calidad>=7,"SI","NO") %>% as.factor()) %>% 
  select(-calidad)

redF = wineF %>% filter(clase=="RED") %>% select(-clase)
whiteF = wineF %>% filter(clase=="WHITE") %>% select(-clase)

summary(redF)

summary(whiteF)

IndexF = createDataPartition(redF$excelencia, p = 0.85, list = F)
redtrainF = redF[IndexF,]
redtestF = redF[-IndexF,]

IndexF = createDataPartition(whiteF$excelencia, p = 0.85, list = F)
whitetrainF = whiteF[IndexF,]
whitetestF = whiteF[-IndexF,]

# ### 1. Random Forest Vino blanco

set.seed(10)

clusterRandForest = makePSOCKcluster(4)
registerDoParallel(clusterRandForest)

RandForestCrossValWhiteF = train(excelencia~., data = whitetrainF,
                               method = 'rf',
                               metric = "Accuracy",
                               trControl = controlRandForest,
                               tuneGrid = matrizRandForest,
                               preProcess = c("center", "scale"))

stopCluster(clusterRandForest)

plot(RandForestCrossValWhiteF)

RandForestCrossValWhiteF$bestTune

RandForestCrossValWhiteFPredictor = predict(RandForestCrossValWhiteF, whitetestF)
RandForestCrossValWhiteFConfMatrix = confusionMatrix(RandForestCrossValWhiteFPredictor, whitetestF$excelencia)
RandForestCrossValWhiteFConfMatrix

RandForestCrossValWhiteFCF = RandForestCrossValWhiteFConfMatrix$table %>% melt()
ggplot(RandForestCrossValWhiteFCF, aes(Reference, y = Prediction))+
  geom_tile(aes(fill = log(value)), color="black")+
  geom_text(aes(label = value)) + 
  scale_fill_gradient(low = "seagreen2", high = "azure") + 
  labs(title = "Matriz de confusi�n RandForest")



# ### 2. Random Forest Vino tinto

set.seed(10)

clusterRandForest = makePSOCKcluster(4)
registerDoParallel(clusterRandForest)

RandForestCrossValRedF = train(excelencia~., data = redtrainF,
                              method = 'rf',
                              metric = "Accuracy",
                              trControl = controlRandForest,
                              tuneGrid = matrizRandForest,
                              preProcess = c("center", "scale"))

stopCluster(clusterRandForest)

plot(RandForestCrossValRedF)

RandForestCrossValRedF$bestTune

RandForestCrossValRedFPredictor = predict(RandForestCrossValRedF, redtestF)
RandForestCrossValRedFConfMatrix = confusionMatrix(RandForestCrossValRedFPredictor, redtestF$excelencia)
RandForestCrossValRedFConfMatrix

RandForestCrossValRedFCF = RandForestCrossValRedFConfMatrix$table %>% melt()
ggplot(RandForestCrossValRedFCF, aes(Reference, y = Prediction))+
  geom_tile(aes(fill = log(value)), color="black")+
  geom_text(aes(label = value)) + 
  scale_fill_gradient(low = "indianred1", high = "ivory") + 
  labs(title = "Matriz de confusi�n RandForest")


# ### 3. Sumario

rbind(`Vino Blanco` = RandForestCrossValWhiteFConfMatrix$overall %>% round(3),
      `Vino Tinto` = RandForestCrossValRedFConfMatrix$overall %>% round(3))

# Los resultados obtenidos en este modelo simplificado de respuesta binaria han mejorado notablemente; aunque el modelo sigue presentando limitaciones y ha obligado a simplificarlo todo, pero es bastante fiable. Esta simplificaci�n 'ayuda' a mejorar el problema con la predicci�n/clasificaci�n en las regiones perif�ricas de la calidad.


# ##4. CONCLUSI�N

# ###1. DISCUSI�N Y EXPOSICI�N DE CONCLUSIONES

# Las conclusiones tanto para el Vino Blanco como el Vinto Tinto dejan claro que el mejor modelo de aprendizaje para este dataset ha de ser de clasificaci�n, y concretamente el algoritmo Random Forest. Esto es, se trata de un dataset que posee muchas variables con problemas de colinealidad y mucho ruido, baja correlaci�n. 
# El mejor modelo que puede ajustarse a las necesidades de clasificaci�n/regresi�n de un dataset de estas caracter�sticas porque el algoritmo del Random Forest aprovecha al m�ximo el llamado fen�meno "bagging", que es la capacidad para promediar interacciones entre datos ruidosas, y poseen baja parcialidad (concretamente no es nuestro caso con todas las variables, pero s� con las m�s importantes)

# El modelo de clasificaci�n frente al de predicci�n parece m�s adecuado para este dataset, aunque en general se puede decir que los datos del Vino Blanco son m�s equilibrados, y muy probablemente esto se deba a que se posee una cantidad de datos bastante mayor de �ste. Este modelo s�lo se considerar�a suficientemente v�lido para usarlo en toma de decisiones si sirviese exclusivamente a los vinos de calidad media.

# Predicci�n - RandomForest para el Vino Blanco

unlist(RandForestCrossValWhiteEvaluator)

# Clasificaci�n - RandomForest para el Vino Tinto

RandForestCrossValRedConfMatrix$overall %>% round(3)

# Clasificaci�n - RandomForest para excelencia de ambos vinos

rbind(`Vino Blanco` = RandForestCrossValWhiteFConfMatrix$overall %>% round(3),
      `Vino Tinto` = RandForestCrossValRedFConfMatrix$overall %>% round(3))

# ###2. PROPUESTAS DE MEJORA

# #### 2.1. T�CNICAS

  # *Un estudio de los outliers m�s profundo (n�tese que en los an�lisis de varianza, mejorando el modelo de regresi�n lineal, se presentaban observaciones muy dispersas de la nube de puntos), y una limpieza de variables colineales (se han intuido algunas durante el comienzo de esta parte del proyecto), podr�a haber ayudado significativamente a la mejora de los modelos
  # *Un estudio y an�lisis m�s meticuloso acerca de los predictores/clasificadores que presentaban mayor error hubiera ayudado a desentra�ar el problema de la predicci�n/clasificaci�n en las regiones perif�ricas.
  # *Un estudio m�s exhaustivo del balanceo de hiperpar�metros. De hecho, durante el desarrollo del proyecto, la variaci�n de los mismos ha producido cambios muy resaltables en el resultado final de algunos modelos. Un mayor conocimiento de todas las opciones que hay para computar podr�a contribuir de gran grado a su mejora
  # *Una mejora de la capacidad de computaci�n o uso de m�s clusters

# #### 2.2. COMERCIALES

  # *Informarse mejor de todo el abanico de variables presentes en el vino, sus car�cter�sticas e incidencias. Una recogida de datos basados en un estudio especializado y auditado por profesionales en�logos podr�a contribuir a una gran mejora iterativa del modelaje, porque no s�lo servir�a para obtener mejores datos si no para tener c�mo y con qu� contrastarlos a posteriori
  # *Consultar a los clientes sobre sus verdaderas necesidades en la venta. Podr�a darse la situaci�n en la que la prioridad de la respuesta a clasificar/predecir no sea la calidad, o que �sta no se haya recogido/estimado bajo los par�metros precisos. Una mejor comunicaci�n con el cliente podr�a ayudar notablemente a su mejora
  # *En general, solicitar m�s datos a los clientes / proveedores. Una mayor cantidad de datos permitir�a la posibilidad de usar modelos m�s complejos (como modelos de Deep Learning) que permitiera un aprendizaje m�s profundo del comportamiento de los vinos
