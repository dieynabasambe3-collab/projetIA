import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erreur : Impossible de lire l'image depuis la caméra.")
        break

    cv2.imshow("Camera IA", frame)

    if cv2.waitKey(1) == 27:  
        break

camera.release()
cv2.destroyAllWindows()