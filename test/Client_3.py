import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
from mtcnn import MTCNN
import dataset

# Load the saved model
# saved_model_path = "C:/Users/aldri/federatedd/global model/final_global_model.keras"
# loaded_model = tf.keras.models.load_model(saved_model_path)
# loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the saved  local model
saved_model_path = "C:/Users/aldri/federatedd/local model/final_local_model_client3.keras"
loaded_model = tf.keras.models.load_model(saved_model_path)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the test dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\Client_3.npz"
x_train, x_test, y_train, y_test = dataset.load_dataset_from_npz(npz_path, test_size=0.2)

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=5)  # Adjust num_classes accordingly

# Evaluate the model on the test data
loss, accuracy = loaded_model.evaluate(x_test, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
loaded_model.summary()

# Initialize FaceNet model and MTCNN detector
embedder = FaceNet()
detector = MTCNN()

def resize_image(image, max_size=(800, 600)):
    """Resize the image to fit within the max_size, maintaining aspect ratio."""
    img = Image.fromarray(image)
    img.thumbnail(max_size, Image.LANCZOS)
    return img

def load_image():
    global img_path, image_rgb
    img_path = filedialog.askopenfilename()
    if img_path:
        image = cv.imread(img_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert image to RGB format
        resized_img = resize_image(image_rgb)
        img_tk = ImageTk.PhotoImage(resized_img)
        panel.img_tk = img_tk
        panel.config(image=img_tk)

def predict():
    global output_image
    if img_path:
        original_image = np.copy(image_rgb)
        faces = detector.detect_faces(image_rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            cropped_face = original_image[y:y+h, x:x+w]
            resized_face = cv.resize(cropped_face, (160, 160))
            embedding = embedder.embeddings([resized_face])[0]
            input_data = np.expand_dims(embedding, axis=0)
            predictions = loaded_model.predict(input_data)
            predicted_class_index = np.argmax(predictions)
            accuracy = np.max(predictions)
            student_labels = ["Student_1", "Student_2", "Student_3", "Student_4", "Student_5"]
            predicted_class_name = student_labels[predicted_class_index]

            # Apply Gaussian blur to the entire image
            blurred_image = cv.GaussianBlur(original_image, (101, 101), 0)
            
            # Draw the rectangle and text on the blurred image
            cv.rectangle(blurred_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{predicted_class_name} ({accuracy:.2f})"
            cv.putText(blurred_image, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the output image for later use
            output_image = blurred_image

            resized_img = resize_image(blurred_image)
            img_tk = ImageTk.PhotoImage(resized_img)
            panel.img_tk = img_tk
            panel.config(image=img_tk)
        else:
            messagebox.showwarning("Warning", "No faces detected in the image.")
    else:
        messagebox.showwarning("Warning", "Please upload an image first.")

def save_image():
    if output_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv.imwrite(save_path, cv.cvtColor(output_image, cv.COLOR_RGB2BGR))
            messagebox.showinfo("Info", "Image saved successfully.")
    else:
        messagebox.showwarning("Warning", "No image to save. Please predict first.")

# Create the main window
root = tk.Tk()
root.title("Face Recognition GUI")

# Create a panel to display the uploaded image
panel = tk.Label(root)
panel.pack()

# Create buttons to upload an image, predict, and save
btn_load = tk.Button(root, text="Upload Image", command=load_image)
btn_load.pack(side="left", padx=10, pady=10)

btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack(side="left", padx=10, pady=10)

btn_save = tk.Button(root, text="Save Image", command=save_image)
btn_save.pack(side="right", padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
