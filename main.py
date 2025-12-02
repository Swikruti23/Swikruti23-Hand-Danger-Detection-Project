import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Virtual danger line (x coordinate)
danger_line_x = 250

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask to detect skin
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours (hand detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    state = "SAFE"
    color = (0, 255, 0)  # Green

    if len(contours) > 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        hand_center_x = x + w // 2

        # Check distance to virtual danger boundary
        distance = abs(hand_center_x - danger_line_x)

        if distance < 40:
            state = "DANGER"
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "DANGER DANGER !!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        elif distance < 100:
            state = "WARNING"
            color = (0, 255, 255)  # Yellow

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Draw the virtual boundary
    cv2.line(frame, (danger_line_x, 0), (danger_line_x, 480), (255, 0, 0), 2)

    # State display
    cv2.putText(frame, state, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Hand Danger Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC to exit

cap.release()
cv2.destroyAllWindows()

