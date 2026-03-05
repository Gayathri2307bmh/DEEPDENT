try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
    print("Warning: cv2 not available - install opencv-python to enable image processing.")
import torch
from PIL import Image
from torchvision import transforms, models

from utils.tooth_detection import detect_teeth

print("Starting prediction...")

# Define model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 5)  # 5 classes

# Load trained weights
model.load_state_dict(torch.load('dental_model.pth'))
model.eval()

print("Model loaded.")

classes = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_tooth(img):

    img = Image.fromarray(img)

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        _,pred = torch.max(output,1)

    return classes[pred.item()]


opg_path = "USE CASE - 01/Dental OPG images/1.jpg"

print(f"Loading OPG image: {opg_path}")

image, teeth = detect_teeth(opg_path)

print(f"Detected {len(teeth)} teeth.")

teeth = sorted(teeth, key=lambda t: t[1][0])

tooth_id = 1

# Simple FDI numbering: assume left to right, upper and lower
# Upper: 11-18 (left), 21-28 (right)
# Lower: 31-38 (left), 41-48 (right)
# But for simplicity, assign 1-32 sequentially, adjust for upper/lower based on y position

mid_y = image.shape[0] // 2  # approximate mid line

fdi_numbers = []

for tooth, (x, y, w, h) in teeth:
    if y + h // 2 < mid_y:  # upper
        quadrant = 1 if x < image.shape[1] // 2 else 2
        base = 10 if quadrant == 1 else 20
    else:  # lower
        quadrant = 3 if x < image.shape[1] // 2 else 4
        base = 30 if quadrant == 3 else 40
    number = base + (tooth_id % 8) + 1  # simple increment
    fdi_numbers.append(number)
    tooth_id += 1

tooth_id = 1

for (tooth, (x, y, w, h)), fdi in zip(teeth, fdi_numbers):

    label = predict_tooth(tooth)

    text = f"FDI {fdi}: {label}"

    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(image,text,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,255,0),2)

    tooth_id += 1

print("Prediction complete.")

# cv2.imshow("DeepDent AI Diagnosis",image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the result
cv2.imwrite("dental_diagnosis_result.jpg", image)
print("Result saved as dental_diagnosis_result.jpg")