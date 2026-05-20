from ultralytics import YOLO

# Charger votre modèle entraîné
model = YOLO("runs/classify/train/weights/best.pt")

# Tester une image
results = model("test.jpg")

# Afficher les résultats
for r in results:
    print(r.probs)