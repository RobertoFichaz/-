import cv2
import numpy as np

# Cargar los pesos y la configuración de YOLOv3
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Cargar las etiquetas de clases COCO
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Capturar video desde la cámara (puedes cambiar el número 0 por la dirección del archivo de video)
cap = cv2.VideoCapture(0)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()

    # Obtener las dimensiones del frame
    alto, ancho = frame.shape[:2]

    # Crear un blob a partir del frame y establecer la entrada para la red YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtener los nombres de las capas de salida
    nombres_capas_salida = net.getUnconnectedOutLayersNames()

    # Realizar el pase hacia adelante
    outs = net.forward(nombres_capas_salida)

    # Procesar las detecciones
    umbral_confianza = 0.5
    umbral_nms = 0.4

    for out in outs:
        for deteccion in out:
            scores = deteccion[5:]
            id_clase = np.argmax(scores)
            confianza = scores[id_clase]
            if confianza > umbral_confianza:
                centro_x = int(deteccion[0] * ancho)
                centro_y = int(deteccion[1] * alto)
                w = int(deteccion[2] * ancho)
                h = int(deteccion[3] * alto)

                x = int(centro_x - w / 2)
                y = int(centro_y - h / 2)

                # Dibujar el cuadro delimitador y la etiqueta en el frame
                color = (0, 255, 0)  # Color verde para el cuadro delimitador
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                etiqueta = f"{classes[id_clase]}: {confianza:.2f}"
                cv2.putText(frame, etiqueta, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el frame resultante
    cv2.imshow('Detección de objetos con YOLOv3', frame)

    # Salir del bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

##asa