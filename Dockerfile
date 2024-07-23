FROM python:3.8-slim

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY . /app

# Instalar las dependencias
RUN pip install -r requirements.txt  
RUN pip install --upgrade streamlit transformers torch Pillow accelerate

# RUN pip install --upgrade streamlit transformers torch Pillow

#--no-cache-dir

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py"]
