import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
# Entender as coordenadas do MediaPipe
# Entender a normalização e desnormalização do pontos do MediaPipe
# Analisar os olhos (seguindo o artigo)
# ponto dos olhos


# Inicializa o mixer do áudio
pygame.mixer.init()

# Carrega o arquivo do som
pygame.mixer.music.load("alarm.mp3")


# FIXME:Olho esquerdo
p_olho_esq = [385, 380, 387, 373, 362, 263]
# FIXME:Olho direito
p_olho_dir = [160, 144, 158, 153, 33, 133]
# FIXME: soma dos olhos
p_olhos = p_olho_esq + p_olho_dir

#FIXME: Pontos da boca 
p_boca = [82,87,13,14,312,78,308]
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
  
#Função MAR
def calculo_mar(face,p_boca):
    try:
      face = np.array([[coord.x, coord.y]for coord in face])
      face_boca = face[p_boca, :]
      
      mar = (np.linalg.norm(face_boca[0] - face_boca[1])+np.linalg.norm(face_boca[2] - face_boca[3])+np.linalg.norm(face_boca[4]- face_boca[5]))/(2*(np.linalg.norm(face_boca[6] - face_boca[7])))

    except: 
      mar = 0.0
    return mar
   
ear_limiar = 0.31 
mar_limiar = 0.1
dormindo = 0

alarme = False

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
      comprimento ,largura,_=frame.shape
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
             #FIXME: normalização para pixel
             face = face_landmarks.landmark
             for id_coord, coord_xyz in enumerate(face):
               if id_coord in p_olhos:
                 coord_cv=mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y,largura,comprimento)
                 cv2.circle(frame,coord_cv,2,(255,0,0),-1)
               if id_coord in p_boca:
                 coord_cv= mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y,largura,comprimento)
                 cv2.circle(frame,coord_cv,2,(255,0,0)-1)
              
                 
             #FIXME: Chamada do EAR e print
             ear = calculo_ear(face,p_olho_dir,p_olho_esq)
             cv2.rectangle(frame, (0,1),(290,140),(58,58,55),-1)
             cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.9, (255, 255, 255), 2)
             
             # FIXME:Chamada do MAR e print
             mar = calculo_mar(face, p_boca)
             cv2.putText(frame, f"MAR: {round(mar,2)}{'Aberto' if mar >= mar_limiar else  'Fechado' }",(1, 50),
                         cv2.FONT_HERSHEY_DUPLEX,
                         0.9,(255,255,255),2)
             
             # Se a boca estiver aberta (MAR > limiar), toca o som
             if mar >= mar_limiar and not alarme:
                    pygame.mixer.music.play(-1)  # Reproduz o som indefinidamente
                    alarme= True

                # Se a boca estiver fechada, pausa o som
             if mar < mar_limiar and alarme:
                    pygame.mixer.music.stop()  # Para o som
                    alarme = False
             
             # # Verificação ear < ear_limiar
             if ear < ear_limiar:
                t_inicial = time.time() if dormindo == 0 else t_inicial
                dormindo = 1
             if dormindo == 1 and ear >= ear_limiar:
                dormindo = 0
             t_final = time.time()
             tempo = (t_final - t_inicial) if dormindo == 1 else 0.0
             
             cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.9, (255, 255, 255), 2)
             if tempo >= 1.5:
                cv2.rectangle(frame, (30,400),(610,452),(109,233,219),-1)
                cv2.putText(frame, f"Muito tempo com os olhos fechados!", (80, 435),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.85, (58, 58, 55), 1)
      except Exception as e:
         print(e)
       
      finally:
         print("processamento conclúido")
      cv2.imshow('Camera', frame)
      if cv2.waitKey(10) & 0xFF == ord('c'):
              break
cap.release()
cv2.destroyAllWindows()