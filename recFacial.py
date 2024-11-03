from flask import Flask, request, jsonify
import flask_cors
import cv2
import numpy as np

app = Flask(__name__)
flask_cors.CORS(app)

# Carregue seu modelo LBPH aqui
model = cv2.face.LBPHFaceRecognizer_create()
model.read("modelo_lbph.xml")  # Certifique-se de que o caminho esteja correto

def show_image(image):
    # Mostra a imagem em uma janela
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow("Imagem Recebida", image)
    cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
    cv2.destroyAllWindows()  # Fecha a janela

@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se a requisição contém uma imagem
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Lê a imagem da requisição
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Padroniza a imagem
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LANCZOS4)

    # Faz a previsão
    label, confidence = model.predict(image)

    # Verifica o resultado do modelo
    if label == 0:
        return jsonify({
            "label": int(label),
            "confidence": float(confidence),
            "message": f"Não foram identificadas manchas ou acnes na pele. Confiança: {confidence:.2f}%"
        })
    else:
        return jsonify({
            "label": int(label),
            "confidence": float(confidence),
            "message": f"Acne identificada. Confiança: {confidence:.2f}%"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
