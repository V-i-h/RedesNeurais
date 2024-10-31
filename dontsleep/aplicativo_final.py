# Importação do opencv-python
import cv2

# Criar  uma variável para camera
cap = cv2.VideoCapture(0)

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

        
    