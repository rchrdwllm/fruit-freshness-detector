from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")
classNames = model.names

results = model.predict(source="images/rotten_banana2.jpg")


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


while True:
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]
        conf = r.conf[0]
        cls = int(r.cls[0])
        class_name = classNames[cls]

        resized_img = resize_with_aspect_ratio(results[0].orig_img, width=600)

        cv2.rectangle(results[0].orig_img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

        label = f"{class_name} {conf:.2f}"
        cv2.putText(results[0].orig_img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Image", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
