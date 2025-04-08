
#PROYECTO FINAL - MODELOS PREDICTIVOS
#VICTORIA RODRIGUEZ

# Cargar librerías necesarias
library(dplyr)
library(readr)
library(lubridate)
library(ggplot2)
library(randomForest)
library(caret)
library(reshape2)

# Ruta del archivo
ruta <- "/Users/victoriarodriguez/Desktop/Modelos predicrivos/Proyecto final/Ecommerce_Consumer_Behavior_Analysis_Data.csv"
df <- read_csv(ruta)

# Corregir nombres de columnas con caracteres especiales
colnames(df) <- make.names(colnames(df))

# LIMPIEZA DE DATOS

# Eliminar el símbolo de dólar y convertir a numérico
df <- df %>%
	mutate(
		Purchase_Amount = as.numeric(gsub("[$,]", "", Purchase_Amount)),
		Time_of_Purchase = mdy(Time_of_Purchase),
		Discount_Used = as.logical(Discount_Used),
		Customer_Loyalty_Program_Member = as.logical(Customer_Loyalty_Program_Member)
	) %>%
	distinct()

# ANÁLISIS EXPLORATORIO DE DATOS (EDA)

# Verificar valores nulos por columna
cat("Valores nulos por columna:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Estadísticas generales
cat("\nResumen estadístico general:\n")
print(summary(df))

# Estadísticas específicas de Purchase_Amount
mean_purchase <- mean(df$Purchase_Amount, na.rm = TRUE)
median_purchase <- median(df$Purchase_Amount, na.rm = TRUE)
sd_purchase <- sd(df$Purchase_Amount, na.rm = TRUE)

cat("\nEstadísticas de Purchase_Amount:\n")
cat("Media:", round(mean_purchase, 2), "\n")
cat("Mediana:", round(median_purchase, 2), "\n")
cat("Desviación estándar:", round(sd_purchase, 2), "\n")


# VISUALIZACIONES

# Histograma del monto de compra
ggplot(df, aes(x = Purchase_Amount)) +
	geom_histogram(binwidth = 50, fill = "cadetblue", color = "white") +
	labs(title = "Distribución del Monto de Compra", x = "Monto de Compra ($)", y = "Frecuencia") +
	theme_minimal()

# Frecuencia de métodos de pago
ggplot(df, aes(x = Payment_Method)) +
	geom_bar(fill = "cadetblue") +
	labs(title = "Frecuencia de Métodos de Pago", x = "Método de Pago", y = "Cantidad de Compras") +
	theme_minimal()


#SEGMENTACIÓN

# Método de pago por género
ggplot(df, aes(x = Payment_Method, fill = Gender)) +
	geom_bar(position = "dodge") +
	labs(title = "Método de Pago por Género", x = "Método de Pago", y = "Cantidad de Compras") +
	theme_minimal()

# Método de pago por nivel de ingreso
ggplot(df, aes(x = Payment_Method, fill = Income_Level)) +
	geom_bar(position = "dodge") +
	labs(title = "Método de Pago por Nivel de Ingreso", x = "Método de Pago", y = "Cantidad de Compras") +
	theme_minimal()

# Boxplot del monto de compra según método de pago
ggplot(df, aes(x = Payment_Method, y = Purchase_Amount, fill = Payment_Method)) +
	geom_boxplot() +
	labs(title = "Distribución del Monto de Compra por Método de Pago", x = "Método de Pago", y = "Monto de Compra ($)") +
	theme_minimal()

# Proporción de uso de descuentos por método de pago
ggplot(df, aes(x = Payment_Method, fill = Discount_Used)) +
	geom_bar(position = "fill") +
	labs(title = "Proporción de Uso de Descuento por Método de Pago", x = "Método de Pago", y = "Proporción") +
	theme_minimal()


# INTERACCIÓN CONTABLE

# Ingresos totales por categoría de producto
ggplot(df %>% group_by(Purchase_Category) %>% summarise(Ingreso_Total = sum(Purchase_Amount, na.rm = TRUE)),
			 aes(x = reorder(Purchase_Category, -Ingreso_Total), y = Ingreso_Total)) +
	geom_col(fill = "cadetblue") +
	labs(title = "Ingresos por Categoría de Producto", x = "Categoría", y = "Ingresos Totales ($)") +
	theme_minimal() +
	coord_flip()

# Ingresos por método de pago
ggplot(df %>% group_by(Payment_Method) %>% summarise(Ingreso_Total = sum(Purchase_Amount, na.rm = TRUE)),
			 aes(x = Payment_Method, y = Ingreso_Total)) +
	geom_col(fill = "cadetblue") +
	labs(title = "Ingresos por Método de Pago", x = "Método de Pago", y = "Ingresos Totales ($)") +
	theme_minimal()

# Ingresos según uso de descuento
ggplot(df %>% group_by(Discount_Used) %>% summarise(Ingreso_Total = sum(Purchase_Amount, na.rm = TRUE)),
			 aes(x = Discount_Used, y = Ingreso_Total, fill = Discount_Used)) +
	geom_col(show.legend = FALSE) +
	labs(title = "Ingresos según Uso de Descuento", x = "¿Usó Descuento?", y = "Ingresos Totales ($)") +
	theme_minimal()

# Ingresos por tipo de envío
ggplot(df %>% group_by(Shipping_Preference) %>% summarise(Ingreso_Total = sum(Purchase_Amount, na.rm = TRUE)),
			 aes(x = Shipping_Preference, y = Ingreso_Total)) +
	geom_col(fill = "cadetblue") +
	labs(title = "Ingresos por Tipo de Envío", x = "Preferencia de Envío", y = "Ingresos Totales ($)") +
	theme_minimal()


# MODELO PREDICTIVO - RANDOM FOREST

# Preparación de los datos
df_model <- df %>%
	select(-Customer_ID, -Time_of_Purchase, -Location) %>%
	na.omit() %>%
	mutate(across(where(is.character), as.factor), Payment_Method = as.factor(Payment_Method))

# División en entrenamiento y prueba
set.seed(123)
train_index <- createDataPartition(df_model$Payment_Method, p = 0.8, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

# Entrenar modelo Random Forest
modelo_rf <- randomForest(Payment_Method ~ ., data = train_data, importance = TRUE, ntree = 200)

# Predicciones y evaluación
predicciones <- predict(modelo_rf, newdata = test_data)
predicciones <- factor(predicciones, levels = levels(test_data$Payment_Method))
conf_matrix <- confusionMatrix(predicciones, test_data$Payment_Method)
print(conf_matrix)


# VISUALIZACIONES DEL MODELO

# Distribución de métodos de pago predichos
ggplot(data.frame(Predicho = predicciones), aes(x = Predicho)) +
	geom_bar(fill = "#AED9E0") +
	labs(title = "Distribución de Métodos de Pago Predichos", x = "Método de Pago Predicho", y = "Cantidad") +
	theme_minimal()

# Comparación entre valores reales y predichos
ggplot(data.frame(Real = test_data$Payment_Method, Predicho = predicciones),
			 aes(x = Real, fill = Predicho)) +
	geom_bar(position = "dodge") +
	scale_fill_manual(values = c(
		"Credit Card" = "#66c2a5",
		"Debit Card"  = "#fc8d62",
		"Cash"        = "#8da0cb",
		"PayPal"      = "#e78ac3",
		"Other"       = "#a6d854"
	)) +
	labs(title = "Comparación de Valores Reales vs. Predicciones",
			 x = "Método de Pago Real", y = "Cantidad", fill = "Predicción") +
	theme_minimal()


# Tasa de aciertos por clase
aciertos <- predicciones == test_data$Payment_Method
por_clase <- data.frame(Metodo = test_data$Payment_Method, Acierto = aciertos)
tasa_acierto <- por_clase %>% group_by(Metodo) %>% summarise(Exactitud = mean(Acierto) * 100)

ggplot(tasa_acierto, aes(x = reorder(Metodo, Exactitud), y = Exactitud)) +
	geom_col(fill = "#AED9E0") +
	coord_flip() +
	labs(title = "Tasa de Acierto por Método de Pago", x = "Método de Pago", y = "Exactitud (%)") +
	theme_minimal()


# MODELO COMPARATIVO - REGRESIÓN LOGÍSTICA MULTICLASE

# Cargar librería adicional
library(nnet)  # Para multinom (regresión logística multiclase)

# Entrenar modelo de regresión logística
modelo_log <- multinom(Payment_Method ~ ., data = train_data)

# Predicciones
pred_log <- predict(modelo_log, newdata = test_data)

# Evaluación
conf_matrix_log <- confusionMatrix(pred_log, test_data$Payment_Method)
print(conf_matrix_log)


# VISUALIZACIÓN COMPARATIVA DE MODELOS

# Crear tabla de comparación de exactitud
accuracy_rf <- sum(predicciones == test_data$Payment_Method) / nrow(test_data)
accuracy_log <- sum(pred_log == test_data$Payment_Method) / nrow(test_data)

comparacion <- data.frame(
	Modelo = c("Random Forest", "Regresión Logística"),
	Exactitud = c(round(accuracy_rf * 100, 2), round(accuracy_log * 100, 2))
)

# Gráfico comparativo
ggplot(comparacion, aes(x = Modelo, y = Exactitud, fill = Modelo)) +
	geom_col(show.legend = FALSE) +
	labs(title = "Comparación de Exactitud entre Modelos",
			 x = "Modelo Predictivo", y = "Exactitud (%)") +
	theme_minimal() +
	scale_fill_manual(values = c("#AED9E0", "#AED9E0"))

#VISUALIZACIÓN 2: Matriz de Confusión Random Forest (Heatmap)

conf_rf_df <- as.data.frame(conf_matrix$table)
ggplot(conf_rf_df, aes(x = Prediction, y = Reference, fill = Freq)) +
	geom_tile() +
	geom_text(aes(label = Freq), color = "black") +
	scale_fill_gradient(low = "white", high = "#AED9E0") +
	labs(title = "Matriz de Confusión - Random Forest") +
	theme_minimal()


# ANÁLISIS DE SERIES DE TIEMPO

# Asegurar que la fecha esté en formato correcto
df <- df %>% mutate(Fecha = as.Date(Time_of_Purchase))

# Ingresos mensuales
ingresos_mensuales <- df %>%
	mutate(Mes = floor_date(Fecha, "month")) %>%
	group_by(Mes) %>%
	summarise(Ingreso_Total = sum(Purchase_Amount, na.rm = TRUE))

ggplot(ingresos_mensuales, aes(x = Mes, y = Ingreso_Total)) +
	geom_line(color = "#AED9E0", linewidth = 1) +
	geom_point(color = "#AED9E0", size = 2) +
	labs(title = "Ingresos Totales Mensuales", x = "Mes", y = "Ingresos ($)") +
	theme_minimal()

# Compras semanales
compras_semanales <- df %>%
	mutate(Semana = floor_date(Fecha, "week")) %>%
	group_by(Semana) %>%
	summarise(Compras = n())

ggplot(compras_semanales, aes(x = Semana, y = Compras)) +
	geom_line(color = "#AED9E0", linewidth = 1) +
	geom_point(color = "#AED9E0", size = 2) +
	labs(title = "Cantidad de Compras Semanales", x = "Semana", y = "Cantidad de Compras") +
	theme_minimal()

# Métodos de pago por mes (gráfico apilado)
metodo_mes <- df %>%
	mutate(Mes = floor_date(Fecha, "month")) %>%
	group_by(Mes, Payment_Method) %>%
	summarise(Cantidad = n())

ggplot(metodo_mes, aes(x = Mes, y = Cantidad, fill = Payment_Method)) +
	geom_area(position = "stack", alpha = 0.8) +
	scale_fill_manual(values = c(
		"Credit Card" = "#66c2a5",
		"Debit Card"  = "#fc8d62",
		"Cash"        = "#8da0cb",
		"PayPal"      = "#e78ac3",
		"Other"       = "#a6d854"
	)) +
	labs(title = "Métodos de Pago por Mes",
			 x = "Mes",
			 y = "Cantidad de Transacciones",
			 fill = "Método de Pago") +
	theme_minimal()


