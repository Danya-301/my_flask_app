from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import cv2
import os
import requests  # Asegúrate de tener esta importación

# Inicializa la app Flask
app = Flask(__name__)

# Lista de nombres de aves
names = ['Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
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
         'Tiluchi Lomirrufo']

# Ruta del modelo
MODEL_PATH = 'modelo/model_VGG16_v4.keras'

def download_model():
    """Descarga el modelo desde Google Drive si no existe localmente."""
    if not os.path.exists(MODEL_PATH):
        url = 'https://drive.google.com/uc?export=download&id=1WEZ60x_yPY-gPv8ugoq_qDLTSAD541zc'
        print("Descargando modelo...")
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Modelo descargado exitosamente.")

# Descarga el modelo si es necesario
download_model()

# Carga el modelo
model = load_model(MODEL_PATH)

# Ruta de subida de imágenes
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Obtiene la imagen subida
        image = request.files["image"]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        image.save(image_path)

        # Procesa la imagen
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Ajusta al tamaño esperado por VGG16
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Realiza la predicción
        preds = model.predict(img)
        predicted_class_index = np.argmax(preds)
        predicted_class_name = names[predicted_class_index]
        confidence_percentage = preds[0][predicted_class_index] * 100

        # Renderiza el resultado
        return render_template("index.html", 
                               prediction=predicted_class_name, 
                               confidence=f"{confidence_percentage:.2f}")

    # Si es una solicitud GET, renderiza la interfaz inicial
    return render_template("index.html")

# Corre la aplicación
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usar puerto de Render si está disponible
    app.run(host="0.0.0.0", port=port, debug=True)
