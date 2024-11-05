import cv2
import mediapipe as mp
import numpy as np
# Entender as coordenadas do MediaPipe
# Entender a normalização e desnormalização do pontos do MediaPipe
# Analisar os olhos (seguindo o artigo)
# ponto dos olhos
# FIXME:Olho esquerdo
p_olho_esq = [385, 380, 387, 373, 362, 263]
# FIXME:Olho direito
p_olho_dir = [160, 144, 158, 153, 33, 133]

#Função EAR
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    
    return media_ear

  
  



cap = cv2.VideoCapture(0)

# vamos importar soluções
# Agora vamos importar as soluções. 
# Existem duas soluções que são mais interessantes para a nossa 
#aplicação. 
# Nós criaremos um novo código e coletaremos os métodos 
# das soluções contidas no MediaPipe. 
# No caso, é o Face Mesh.
# Primeiro, vamos importar a solução de desenho, 
# para observarmos os pontos na nossa face que o MediaPipe coleta.
# Para isso, precisamos fazer:

mp_drawing = mp.solutions.drawing_utils

#Ainda falta coletar a solução do Face Mesh.
# Para isso, usaremos mp_face_mesh, objeto que receberá a solução, 
# igual a mp.solutions.face_mesh
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
  while cap.isOpened():
      sucesso, frame = cap.read()
      if not sucesso:
        print('Ignorando o frame vazio da câmera.')
        continue
      # transformando de BGR para RGB
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # variável que vai receber os dados processados do meu frame, como os pontos do meu rosto etc. 
      saida_facemesh = facemesh.process(frame)
      # transformar novamente para BGR, já que o OpenCV trabalha com BGR
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      # mostrar os pontos da nossa face
      # o process podemos processar os pontos do nosso rosto
      # agora vamos poder mostrar esses pontos
      # vamos mostrar essa detecção que o mediapipe fez
      # face_landmars - coordenadas da nossa face
      # percorrendo nosso processamento
      #FIXME: acesso as coordenadas (multi_face_landmarks)
      try:
          for face_landmarks in saida_facemesh.multi_face_landmarks:
             """
               Chamamos o objeto mp_drawing e utilizamos o método draw_landmarks()
               para o desenho de cada ponto/coordenada que for coletada do nosso rosto.
               
               Dentro dos parênteses, colocaremos o frame, que é o que está sendo coletado,
               e o face_landmarks, que são as coordenadas de cada ponto.
               Ainda nos parênteses, utilizaremos o mp_face_mesh.FACEMESH_CONTOURS para especificar 
               os nossos pontos
               
             """
             mp_drawing.draw_landmarks(frame, 
                                       face_landmarks, 
                                       mp_face_mesh.FACEMESH_CONTOURS,
                                       landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255,102,102),thickness=1,circle_radius=1),
                                        connection_drawing_spec = mp_drawing.DrawingSpec(color=(102,204,0),
                                                                                         thickness=1,
                                                                                         circle_radius=1))
      except Exception as e:
         print(e)
       
      finally:
         print("processamento conclúido")
      cv2.imshow('Camera', frame)
      if cv2.waitKey(10) & 0xFF == ord('c'):
              break
cap.release()
cv2.destroyAllWindows()