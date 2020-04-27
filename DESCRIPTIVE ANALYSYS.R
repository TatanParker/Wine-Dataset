# Integración de librerías

rm(list=ls()) ###Limpiamos el entorno global

# Instalación de paquetes y declaración de librerías

packages = c("tidyverse", "RCurl", "psych", "stats", 
             "randomForest", "glmnet", "caret","kernlab", 
             "rpart", "rpart.plot", "neuralnet", "C50",
             "doParallel", "AUC", "ggfortify", "rmdformats", 
             "corrgram", "ggplot2", "naniar", "e1071",
             "lattice", "caret", "knitr", "kable", "corrplot", 
             "kknn", "randomForest", "kernlab", "car", "xlsx", 
             "data.table", "GGally", "gplots", "kableExtra")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
invisible(lapply(packages, require, character.only = TRUE))

# Creacion de una funcion que elimina outliers en función de los valores que se encuentran dentro de los percentiles altos

remove_outliers <- function(x,quant){
  
  require("dplyr")
  
  for(i in 1:ncol(x)){
    
    x = mutate(x,outliers=ifelse(x[[i]] < quantile(x[[i]],quant),0,1))
    x = filter(x, outliers==0) 
    x = select(x, -outliers)
    
  }
  
  return(x)
  
}

# ### Carga de los datos:

setwd("D:/MAIN/MASTER/M11/WINE")


wine = data.frame(read.xlsx("SOURCE/SOURCE.xlsx", sheetIndex = 1))

  NewNames = c("clase","acidez_fija","acidez_volatil","acidez_citrica","azucar_res","cloruros","sulfitos_libres","sulfitos_totales","densidad","ph","sulfatos","alcohol","calidad")
  
  names(wine)=NewNames

# ### Limpieza y preparación de datos

# Chequeamos los NA del dataset
  
vis_miss(wine)
  
    ###Ninguno

head(wine)

summary(wine)

# Se observan valores relativamente esperables, según los valores comprendidos como "usuales" según el estudio preliminar que hemos realizado. Valores importantes como la densidad, alcohol, pH, sulfitos totales, cloruros o acidez se encuentran dentro del rango de valores esperado.
# También puede observarse que hay máximos que muestran valores fuera de los percentiles altos y que se candidatan como **outliers**, como es el caso de los sulfitos libres, totales, y el azucar residual.
# En general, el set de datos está bastante limpio y apenas requiere de limpieza, aparte de los outliers, que no se eliminarán aun hasta hacer el análisis exploratorio, a excepción de los más fuertes. Se procederá a la división por tipo de vino.

str(wine)
  
  #Se confirman que sólo el atributo clase, agregado al dataset original como factor



# Se crean pues, los datasets de las dos clases de vino por separado, eliminando el atributo clase y eliminando outliers fuertes:
 
white = wine %>% filter(clase=="WHITE") %>% select(-clase)

  white = remove_outliers(white, 0.999) 

  ##Eliminamos los outliers más fuertes (por encima del percentil 99.9)

red = wine %>% filter(clase=="RED") %>% select(-clase)

  red = remove_outliers(red, 0.999) 

# Analisis Descriptivo
  
# Mapa de correlaciones de ambos datasets
    
  cor_white = corrgram(white, type="data",
           lower.panel=panel.conf, 
           upper.panel=panel.shade,
           main= "Mapa de correlaciones del vino blanco",
           order=T,
           cex.labels=1.2,
           col.regions = colorRampPalette(c("darkgoldenrod4", "burlywood1",
                                            "darkkhaki", "darkgreen")))
  
  cor_red = corrgram(red, type="data",
           lower.panel=panel.conf, 
           upper.panel=panel.shade,
           main= "Mapa de correlaciones del vino tinto",
           order=T,
           cex.labels=1.2,
           col.regions = colorRampPalette(c("yellow", "salmon", "blue"))) 

  
  correlation_white = round(as.data.frame(cor(as.matrix(white))),2)
  
  correlation_white[abs(correlation_white)<0.4] = "*"
  
  correlation_white = correlation_white %>% 
    kable() %>% kable_styling(bootstrap_options = "striped", full_width = F) %>% 
    add_header_above(c(" ", "Vino blanco" = 12)) 
  add_header_above(kable(correlation_white,"html"), header = "hola")
  
# Vino Blanco
   
# Los predictores de azúcar, pH y ácido cítrico no juegan un papel  **aparentemente** relevante en la calidad del vino.
# Las correlaciones son débiles entre la calidad y el ácido cítrico, los sulfitos libres, así como los sulfatos.
  
# La densidad tiene una correlación de 0.83 con el azúcar residual y una correlación de -0.81 con alcohol. El alcohol es el único predictor que está considerablemente relacionado con la calidad del vino.

  correlation_red = round(as.data.frame(cor(as.matrix(red))),2)
  
  correlation_red[abs(correlation_red)<0.4] = "*"
  
  correlation_red %>% 
    kable() %>% kable_styling(bootstrap_options = "striped", full_width = F) %>% 
    add_header_above(c(" ", "Vino tinto" = 12)) 
  add_header_above(kable(correlation,"html"), header = "hola")
  
# Vino tinto

# Entre los dos vinos se dan algunas similitudes importantes, como es la correlación calidad-alcohol, alcohol-densidad(aunque con bastante menos fuerza), acidez-fija-pH, pero es de esperar pues se trata, como se explicaba en la introducción, de variables físico-químicamente dependientes. Sin embargo en el vino tinto se dan correlaciones únicas, como son la relación entre la acidez fija y volátil con la acidez cítrica.
# De hecho, la acidez cítrica tiene especial relevancia en el vino tinto, ya que está también está fuertemente ligada al pH. Los sulfitos, en cambio, pierden importancia, y la acidez fija se correlaciona con fuerza con la densidad.
  
# En el análisis exploratorio se ahondará en estas cuestiones.
 
# Usaremos maravillosa función llamada pairs.panel de la librería psych, que genera un plot con un mapa completo de SPLOM, correlaciones e histogramas de las variables continuas
  
pairs.panels(white)

pairs.panels(red)
