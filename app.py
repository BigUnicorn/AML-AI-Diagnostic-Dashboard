import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
MODEL_PATH = 'aml_model.h5'
IMG_SIZE = (144, 144)

# TERMİNAL ÇIKTINA GÖRE SIRALAMA (Classification Report sırası)
CLASS_NAMES = ['CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1', 'control']

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_aml_model():
    try:
        # Önce tüm modeli yüklemeyi dene
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except:
        # Hata verirse mimariyi kurup ağırlıkları yükle (TypeError çözümün için)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(144, 144, 3)),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        model.load_weights(MODEL_PATH)
        return model

model = load_aml_model()

@app.route('/')
def index():
    # Terminal çıktındaki 'support' değerlerine göre yaklaşık toplam dağılım
    stats = {
        "labels": ["CBFB_MYH11", "NPM1", "PML_RARA", "Control", "RUNX1_T1"],
        "counts": [17260, 17740, 11210, 14900, 20030] # 81k toplam için oranlandı
    }
    # Terminaldeki Classification Report değerlerin
    metrics = {
        "accuracy": 0.59, 
        "f1_control": 0.82,
        "f1_weighted": 0.58
    }
    return render_template('index.html', stats=stats, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Görüntü hazırlama (144x144)
    img = tf.keras.utils.load_img(filepath, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    conf = 100 * np.max(preds[0])
    
    descriptions = {
        'CBFB_MYH11': "AML with CBFB::MYH11 fusion. WHO 2022 sınıflamasına göre genetik anomali tespit edildi.",
        'NPM1': "AML with NPM1 mutation. En yaygın AML mutasyonlarından biridir.",
        'PML_RARA': "APL with PML::RARA fusion. Acil müdahale gerektiren sitomorfolojik yapı.",
        'control': "Normal/Sağlıklı hücre yapısı. Herhangi bir lösemik anomali saptanmadı.",''
        'RUNX1_RUNX1T1': "AML with RUNX1::RUNX1T1 fusion. Belirgin morfolojik granülasyonlar içerebilir."


    }
        
    
    return jsonify({
        "class": CLASS_NAMES[idx],
        "confidence": f"{conf:.2f}%",
        "description": descriptions.get(CLASS_NAMES[idx], "Analiz tamamlandı.")
    })

if __name__ == '__main__':
    app.run(debug=True)