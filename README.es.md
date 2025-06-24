# Creo mi entorno virtual para trabajar mejor y evitar mezclar versiones de librerias con python que está ne local. 

Mi entorno virtual es: .\env\Scripts\activate.bat

pip install -r requirements.txt


# Proyecto de Detección de Personas con YOLOv8

Este proyecto implementa un modelo YOLOv8 para detección de personas utilizando un dataset personalizado con imágenes reescaladas y aumentadas mediante técnicas de procesamiento de imágenes.

---

## 1. Preprocesamiento de imágenes: Reescalado y aumentos

El dataset original contiene imágenes de tamaños variados y relativamente pocas muestras. Para mejorar la calidad y cantidad de datos:

- **Se reescalan todas las imágenes a 640x640 píxeles**, asegurando una entrada uniforme para el modelo.
- Se aplican **aumentos de datos** como rotaciones aleatorias, cambios en contraste, brillo, desenfoque, y volteos horizontales para enriquecer la variedad del dataset.
- Esto ayuda a que el modelo generalice mejor y evite sobreajuste.

---

## 2. Conversión de anotaciones (.odgt a .txt formato YOLO)

Las anotaciones originales están en formato `.odgt`. Luego de reescalar y aplicar aumentos, es necesario:

- Ajustar las coordenadas de las cajas delimitadoras a las nuevas dimensiones (640x640).
- Convertir estas anotaciones al formato `.txt` compatible con YOLO, que usa coordenadas normalizadas y el formato:  
  `class x_center y_center width height`
- Para cada imagen aumentada se copia o adapta su correspondiente archivo `.txt` para mantener la correspondencia.

---

## 3. Estructura del proyecto

/data

/Train # Imágenes originales de entrenamiento

/Train640 # Imágenes reescaladas a 640x640

/Validation # Imágenes originales de entrenamiento

/Validation640 # Imágenes reescaladas a 640x640

/labels # Archivos .txt con las anotaciones para YOLO

annotation_train.odgt # Archivo original con anotaciones en formato ODGT

/runs # Carpeta generada con resultados y pesos entrenados

entrenar_yolo.py # Script principal para entrenamiento
preprocesamiento.py # Script para reescalar y aumentar imágenes + convertir anotaciones

---

## 4. Archivo YAML para configuración de datos

Se crea un archivo `.yaml` para definir la configuración del dataset para YOLO, por ejemplo `person_data.yaml`:

```yaml
train: ../data/Train640_aug
val: ../data/Val640_aug

nc: 1  # Número de clases (1 clase: persona)
names: ['person']