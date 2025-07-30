# Cargar las librerías necesarias
import cv2
import os

# Ruta donde se almacenan las imágenes de entrenamiento
dataPath = './Data'
# Lista de imágenes en el directorio Data
imagePaths = os.listdir(dataPath)
# Imprimir las rutas de las imágenes
print('imagePaths=',imagePaths)

# Creando el reconocedor de rostros
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Leyendo el modelo
face_recognizer.read('./modeloEigenFace.xml')
# Cargando el video
cap = cv2.VideoCapture("./Muestras/pdvm.mp4")
# Cargando el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Bucle para capturar fotogramas del video
while True:
    # Capturando fotogramas del video
	ret,frame = cap.read()
 	# Si no hay fotogramas, salir del bucle
	if ret == False: break
	# Convertir el fotograma a escala de grises
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Crear una copia del fotograma original para dibujar los rectángulos
	auxFrame = gray.copy()
	# Detectar rostros en el fotograma
	faces = faceClassif.detectMultiScale(gray,1.3,5)
	# Si no se detectan rostros, continuar al siguiente fotograma
	for (x,y,w,h) in faces:
		# Dibujar un rectángulo alrededor del rostro detectado
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)
		# Dibujar un rectángulo alrededor del rostro detectado
		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		# EigenFaces
		if result[1] < 5700:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
	# Mostrar el fotograma con los rostros detectados y reconocidos		
	cv2.imshow('frame',frame)
	# Esperar 1 ms para continuar al siguiente fotograma
	k = cv2.waitKey(1)
	# Si se presiona la tecla 'ESC', salir del bucle
	if k == 27: break

# Liberar el video y cerrar las ventanas
cap.release()
# Cerrar todas las ventanas
cv2.destroyAllWindows()