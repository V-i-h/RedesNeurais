# Importação do opencv-python
import cv2
import mediapipe as mp 
# Criar  uma variável para camera
cap = cv2.VideoCapture(0)
# Usando uma solução de desenho
mp_drawing = mp.solutions.drawing_utils
# usando uma solução para Face Mesh Detection
mp_face_mesh = mp.solutions.face_mesh
# Liberação automática
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    # Enquanto a camera estiver aberta
    while cap.isOpened():
        # sucesso - booleana (Verificar se o frame está vazio) 
        # frame - captura
        sucesso, frame = cap.read()
        # realizar a verificação
        # sucesso = 1 fracasso = 0
        if not sucesso :
            print('Ignorando o frame vazio da camêra')
            continue
        # Transformando de BGR para RGB
        frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Criar uma variável dados processados(ex: pontos do rosto)
        saida_facemesh = facemesh.process(frame)
        # O OpenCv- entende BGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        """
            1- Mostrar os pontos da nossa face
            2- O process - processar os dados
            3- face_landmarks(COORDENADAS)
            
        """
        for face_landmarks in saida_facemesh.multi_face_landmarks:
            # desenhar
            mp_drawing.draw_landmarks(frame,face_landmarks,mp_face_mesh.FACEMESH_CONTOURS)
            
        
        # Carregar frame - com título
        cv2.imshow ("Camêra",frame)
        # bitwise  tabela ASC II
        # 10 milissegundos
        # ord() - retorna o valor Unicode (ou ASC II)
        # o valor 0xFF é tabela ASC II  estendida
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break
        
cap.release()
cv2.destroyAllWindows()

        
    