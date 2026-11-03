import tkinter as tk
import webbrowser
from tkinter import filedialog, Label, Button, Canvas, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import pandas as pd
import pyttsx3
import text2speech as ts
# --- YOLOv8 Specific Imports ---
from ultralytics import YOLO

df = pd.read_csv('data.csv')

# --- YOLOv8 Model Loading ---
MODEL_PATH = 'best.pt'
try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model loaded from {MODEL_PATH}")
except Exception as e:
    messagebox.showerror("Model Error",
                         f"Error loading YOLOv8 model: {e}\nPlease ensure '{MODEL_PATH}' is in the same directory and 'ultralytics' is installed.")
    exit()

try:
    label_names_dict = yolo_model.names
    label_names = [label_names_dict[i] for i in sorted(label_names_dict.keys())]
    if not label_names:
        raise ValueError("YOLO model class names could not be loaded or are empty.")
    print("Class names loaded:", label_names)
except Exception as e:
    messagebox.showerror("Label Error", f"Error getting class names from YOLO model: {e}")
    exit()

# Ensure the 'output' directory exists
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")


class ImagePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Predictor (YOLOv8)")
        self.root.geometry("1100x750")
        self.root.config(bg="#f0f0f0")

        self.image_path = None
        self.yolo_model = yolo_model
        self.label_names = label_names

        # Store image references to prevent garbage collection
        self.photo_input = None
        self.photo_result = None

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

        # --- Header ---
        self.header_frame = tk.Frame(root, bg="#000066", height=60)
        self.header_frame.pack(fill="x", pady=10)
        self.header_label = Label(self.header_frame, text="Gemstone detection using (YOLOv8)",
                                  font=("Arial", 24, "bold"), bg="#000066", fg="white")
        self.header_label.pack(pady=10)

        # --- Main Content Frame ---
        self.main_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Left side for input image display and buttons
        self.left_frame = tk.Frame(self.main_frame, bg="#ffffff", width=500)
        self.left_frame.pack(side="left", padx=20, pady=20, fill="y")
        self.left_frame.pack_propagate(False)

        # Input Image Canvas: 400x400
        self.input_label_title = Label(self.left_frame, text="Uploaded Image", font=("Arial", 12, "bold"), bg="#ffffff")
        self.input_label_title.pack(anchor="w")

        self.image_canvas = Canvas(self.left_frame, width=400, height=400, bg="#e0e0e0", bd=2, relief="solid")
        self.image_canvas.pack(pady=10)
        self.placeholder_label = Label(self.image_canvas, text="No Image Selected", font=("Arial", 12), fg="gray",
                                       bg="#e0e0e0")
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

        # Buttons
        self.upload_button = Button(self.left_frame, text="Upload Image", command=self.upload_image, width="20",
                                    font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                                    activebackground="#1976D2", relief="raised", bd=3)
        self.upload_button.place(relx=0.10, rely=0.82)

        self.predict_button = Button(self.left_frame, text="Predict", command=self.predict_image, width="20",
                                     font=("Arial", 10, "bold"), bg="#FFC107", fg="black",
                                     activebackground="#FFA000", relief="raised", bd=3, state=tk.DISABLED)
        self.predict_button.place(relx=0.50, rely=0.82)

        self.close_button = Button(self.left_frame, text="Close",
                                   command=self.close_app,
                                   font=("Arial", 14, "bold"),
                                   bg="#F44336", fg="white",
                                   activebackground="#D32F2F",
                                   relief="raised", bd=3)
        self.close_button.pack(side="bottom", pady=10, ipadx=10, ipady=5)

        # Right side for results & annotated image canvas
        self.right_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.right_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)

        self.result_label = Label(self.right_frame, text="Prediction Result:",
                                  font=("Arial", 16, "bold"), bg="#ffffff", fg="#333333")
        self.result_label.pack(pady=(0, 5), anchor="w")

        self.confidence_label = Label(self.right_frame, text="Confidence: N/A",
                                      font=("Arial", 14), bg="#ffffff", fg="#555555")
        self.confidence_label.pack(pady=(0, 10), anchor="w")

        # Result Annotated Image Canvas (Replaces the matplotlib plot)
        self.result_label_title = Label(self.right_frame, text="Annotated Result", font=("Arial", 12, "bold"),
                                        bg="#ffffff")
        self.result_label_title.pack(anchor="w")

        self.result_canvas = Canvas(self.right_frame, width=400, height=400, bg="#e0e0e0", bd=2, relief="solid")
        self.result_canvas.pack(pady=5)
        self.result_placeholder = Label(self.result_canvas, text="Result Will Appear Here", font=("Arial", 12),
                                        fg="gray",
                                        bg="#e0e0e0")
        self.result_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def botsays(self, message):
        engine = pyttsx3.init()
        engine.say("the prediction is " + message)
        engine.runAndWait()

    def display_info(self, item):
        try:
            if item in df['name'].values:
                row = df[df['name'] == item].iloc[0]
                infomessage = f"Gemstone Name: {row['name']}\n\nGemstone Info: {row['info']}"
                messagebox.showinfo("Recommendation", infomessage)
                ts.botsay(infomessage)
            else:
                messagebox.showinfo("Information", f"No information available for {item}")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching information:\n{e}")

    def close_app(self):
        self.speak_text("Closing the application. Goodbye!")
        self.root.after(1000, self.root.destroy)

    def webcamlive(self):
        os.system("python liveWebcam_detection.py")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((400, 400), Image.LANCZOS)
            self.photo_input = ImageTk.PhotoImage(img)

            # Draw input image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo_input)

            # Reset UI states
            self.predict_button.config(state=tk.NORMAL)
            self.result_label.config(text="Prediction Result:")
            self.confidence_label.config(text="Confidence: N/A")
            self.clear_result_canvas()

    def clear_result_canvas(self):
        self.result_canvas.delete("all")
        self.result_placeholder = Label(self.result_canvas, text="Result Will Appear Here", font=("Arial", 12),
                                        fg="gray", bg="#e0e0e0")
        self.result_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def predict_image(self):
        if not self.image_path:
            return

        results = self.yolo_model.predict(source=self.image_path, save=False, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        boxes = result.boxes

        predicted_label = "no image"
        confidence = 0.0
        output_image_path = os.path.join(OUTPUT_DIR, "result_img.jpg")

        if not boxes or len(boxes) == 0:
            self.result_label.config(text=f"Prediction Result: No object detected")
            self.confidence_label.config(text=f"Confidence: N/A")

            # Show clear placeholder if nothing detected
            self.clear_result_canvas()

            if os.path.exists(output_image_path):
                os.remove(output_image_path)
        else:
            max_conf_idx = boxes.conf.argmax().item()
            predicted_class_index = int(boxes.cls[max_conf_idx].item())
            confidence = boxes.conf[max_conf_idx].item() * 100
            predicted_label = self.label_names[predicted_class_index]

            self.result_label.config(text=f"Prediction Result: {predicted_label}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            try:
                # Render annotations to image array
                annotated_img_np = result.plot()
                annotated_img_pil = Image.fromarray(annotated_img_np[..., ::-1])  # BGR to RGB
                annotated_img_pil.save(output_image_path)

                # Load and display in the right canvas
                img_to_display = Image.open(output_image_path)
                img_to_display = img_to_display.resize((400, 400), Image.LANCZOS)
                self.photo_result = ImageTk.PhotoImage(img_to_display)

                self.result_canvas.delete("all")
                self.result_canvas.create_image(0, 0, anchor="nw", image=self.photo_result)

            except Exception as e:
                print(f"Error displaying annotated image: {e}")

        self.display_info(predicted_label)

    def speak_text(self, text):
        self.root.after(100, lambda: self._speak_thread_safe(text))

    def _speak_thread_safe(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()