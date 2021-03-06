# Integraci�n de librer�as

rm(list=ls()) ###Limpiamos el entorno global

# Instalaci�n de paquetes y declaraci�n de librer�as

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

# Creacion de una funcion que elimina outliers en funci�n de los valores que se encuentran dentro de los percentiles altos

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

# ### Limpieza y preparaci�n de datos

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
 
white = wine %>% filter(clase=="WHITE") %>% select(-clase)

  white = remove_outliers(white, 0.999) 

  ##Eliminamos los outliers m�s fuertes (por encima del percentil 99.9)

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
   
# Los predictores de az�car, pH y �cido c�trico no juegan un papel  **aparentemente** relevante en la calidad del vino.
# Las correlaciones son d�biles entre la calidad y el �cido c�trico, los sulfitos libres, as� como los sulfatos.
  
# La densidad tiene una correlaci�n de 0.83 con el az�car residual y una correlaci�n de -0.81 con alcohol. El alcohol es el �nico predictor que est� considerablemente relacionado con la calidad del vino.

  correlation_red = round(as.data.frame(cor(as.matrix(red))),2)
  
  correlation_red[abs(correlation_red)<0.4] = "*"
  
  correlation_red %>% 
    kable() %>% kable_styling(bootstrap_options = "striped", full_width = F) %>% 
    add_header_above(c(" ", "Vino tinto" = 12)) 
  add_header_above(kable(correlation,"html"), header = "hola")
  
# Vino tinto

# Entre los dos vinos se dan algunas similitudes importantes, como es la correlaci�n calidad-alcohol, alcohol-densidad(aunque con bastante menos fuerza), acidez-fija-pH, pero es de esperar pues se trata, como se explicaba en la introducci�n, de variables f�sico-qu�micamente dependientes. Sin embargo en el vino tinto se dan correlaciones �nicas, como son la relaci�n entre la acidez fija y vol�til con la acidez c�trica.
# De hecho, la acidez c�trica tiene especial relevancia en el vino tinto, ya que est� tambi�n est� fuertemente ligada al pH. Los sulfitos, en cambio, pierden importancia, y la acidez fija se correlaciona con fuerza con la densidad.
  
# En el an�lisis exploratorio se ahondar� en estas cuestiones.
 
# Usaremos maravillosa funci�n llamada pairs.panel de la librer�a psych, que genera un plot con un mapa completo de SPLOM, correlaciones e histogramas de las variables continuas
  
pairs.panels(white)

pairs.panels(red)
