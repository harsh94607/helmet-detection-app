# import cv2
# import torch

# # Load your trained model
# model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

# # Use working webcam index (1 or 2 based on your test)
# cap = cv2.VideoCapture(2)

# print("🎥 Starting real-time helmet detection. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     # Run detection
#     results = model(frame)
#     results.render()

#     # Show result
#     cv2.imshow("Helmet Detection", results.ims[0])

#     # Quit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()








import cv2
import torch

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
cap = cv2.VideoCapture(2)  # Change to 2 if needed

print("🟢 Starting real-time helmet detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results.render()

    cv2.imshow("Helmet Detection", results.ims[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
