from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import cv2
import os

# Inicializa la app Flask
app = Flask(__name__)

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Lista de nombres de aves (asegúrate de que los índices coincidan con el modelo)
names = [
    'Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
    'Aratinga de Pinceles', 'Aratinga de Wagler', 'Aratinga Ojiblanca', 'Aratinga Orejigualda',
    'Aratinga Pertinaz', 'Batará Barrado', 'Batará Crestibarrado', 'Batara Crestinegro',
    'Batará Mayor', 'Batará Pizarroso Occidental', 'Batará Unicolor', 'Cacatua Ninfa',
    'Catita Frentirrufa', 'Cotorra Colinegra', 'Cotorra Pechiparda', 'Cotorrita Alipinta',
    'Cotorrita de Anteojos', 'Guacamaya Roja', 'Guacamaya Verde', 'Guacamayo Aliverde',
    'Guacamayo azuliamarillo', 'Guacamayo Severo', 'Hormiguerito Coicorita Norteño',
    'Hormiguerito Coicorita Sureño', 'Hormiguerito Flanquialbo', 'Hormiguerito Leonado',
    'Hormiguerito Plomizo', 'Hormiguero Azabache', 'Hormiguero Cantor', 'Hormiguero de Parker',
    'Hormiguero Dorsicastaño', 'Hormiguero Guardarribera Oriental', 'Hormiguero Inmaculado',
    'Hormiguero Sencillo', 'Hormiguero Ventriblanco', 'Lorito Amazonico', 'Lorito Cabecigualdo',
    'Lorito de fuertes', 'Loro Alibronceado', 'Loro Cabeciazul', 'Loro Cachetes Amarillos',
    'Loro Corona Azul', 'Loro Tumultuoso', 'Ojodefuego Occidental', 'Periquito Alas Amarillas',
    'Periquito Australiano', 'Periquito Barrado', 'Tiluchí Colilargo', 'Tiluchí de Santander',
    'Tiluchi Lomirrufo'
]

# Carga el modelo una vez para optimizar memoria
try:
    model = load_model('modelo/model_VGG16_v4.keras')
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Obtiene la imagen subida
        image = request.files.get("image")
        
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            image.save(image_path)

            # Procesa la imagen
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (224, 224))  # Ajusta al tamaño esperado por VGG16
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)

                # Realiza la predicción
                preds = model.predict(img)
                predicted_class_index = np.argmax(preds)

                # Verifica que el índice esté en rango
                if 0 <= predicted_class_index < len(names):
                    predicted_class_name = names[predicted_class_index]
                    confidence_percentage = preds[0][predicted_class_index] * 100
                else:
                    predicted_class_name = "Clase desconocida"
                    confidence_percentage = 0.0

                # Renderiza el resultado
                return render_template(
                    "index.html", 
                    prediction=predicted_class_name, 
                    confidence=f"{confidence_percentage:.2f}"
                )
            except Exception as e:
                return render_template(
                    "index.html", 
                    prediction=f"Error en la predicción: {str(e)}", 
                    confidence="0.00"
                )

    # Si es una solicitud GET, renderiza la interfaz inicial
    return render_template("index.html")

# Ejecuta la aplicación
if __name__ == "__main__":
    app.run(debug=False)