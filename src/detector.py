import cv2
import torch
import time

# Carrega o modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)  # '0' usa a câmera padrão

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Executa a detecção no frame capturado
    results = model(frame)

    # Exibe os resultados na tela
    results.render()  # Para desenhar as detecções no frame

    # Exibe a imagem com as detecções
    cv2.imshow('Detecção de Objetos', frame)

    # Captura a imagem e salva quando pressionar 'S'
    if cv2.waitKey(1) & 0xFF == ord('s'):  # 's' para salvar a imagem
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"captura_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Imagem salva como {filename}")

    # Para encerrar, pressione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()
