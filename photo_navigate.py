from PIL import Image, ImageTk, ImageDraw
from natsort import natsorted
import os
import tkinter as tk

class ImageBrowser:
    def __init__(self, folder_path):
        self.image_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        if not self.image_files:
            print("Brak obrazów w folderze.")
            return

        self.folder_path = folder_path
        self.max_width = 800
        self.max_height = 600
        self.image_index = 0

        self.root = tk.Tk()
        self.root.title("Przeglądanie obrazów")
        self.current_label = None
        self.rect_start = None
        self.drawing_rect = False

        self.setup_ui()

    def setup_ui(self):
        self.show_image(self.image_index)

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.previous_image)
        self.root.mainloop()

    def resize_image(self, image, max_width, max_height):
        width, height = image.size
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height), Image.ANTIALIAS)
        return image

    def show_image(self, index):
        if self.current_label:
            self.current_label.destroy()

        image_file = self.image_files[index]
        image_path = os.path.join(self.folder_path, image_file)
        image = Image.open(image_path)
        image = self.resize_image(image, self.max_width, self.max_height)
        self.current_image = image
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(self.root, image=photo)
        label.image = photo
        label.pack()

        self.current_label = label

        self.current_label.bind("<ButtonPress-1>", self.start_rect)
        self.current_label.bind("<B1-Motion>", self.draw_rect)
        self.current_label.bind("<ButtonRelease-1>", self.end_rect)

    def next_image(self, event):
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.show_image(self.image_index)

    def previous_image(self, event):
        self.image_index = (self.image_index - 1) % len(self.image_files)
        self.show_image(self.image_index)

    def start_rect(self, event):
        self.rect_start = (event.x, event.y)
        self.drawing_rect = True

    def draw_rect(self, event):
        if self.drawing_rect:
            x0, y0 = self.rect_start
            x1, y1 = event.x, event.y
            image_with_rect = self.current_image.copy()
            draw = ImageDraw.Draw(image_with_rect)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            photo = ImageTk.PhotoImage(image_with_rect)

            self.current_label.configure(image=photo)
            self.current_label.image = photo

    def end_rect(self, event):
        if self.drawing_rect:
            self.drawing_rect = False
            img_width, img_height = self.current_image.size
            x0, y0 = self.rect_start
            x1, y1 = event.x, event.y
            x0_percent = x0 / img_width
            y0_percent = y0 / img_height
            x1_percent = x1 / img_width
            y1_percent = y1 / img_height
            coordinates = f"0 {x0_percent:.6f} {y0_percent:.6f} {x1_percent:.6f} {y1_percent:.6f}"
            print(f"Zaznaczono prostokąt: {coordinates}")
            self.save_coordinates(coordinates)

    def save_coordinates(self, coordinates):
        dataset_path = os.path.dirname(self.folder_path)
        labels_folder = os.path.join(dataset_path, "labels")
        if not os.path.exists(labels_folder):
            os.mkdir(labels_folder)
        filename = os.path.join(labels_folder, f"{os.path.splitext(self.image_files[self.image_index])[0]}.txt")
        with open(filename, "w") as file:  # Użyj "w" zamiast "a" aby nadpisywać plik, nie dodawać do niego
            file.write(coordinates + "\n")

if __name__ == "__main__":
    folder_path = r"C:\Users\mateu\Desktop\wykrywanie_rzeczy\dataset\images"
    ImageBrowser(folder_path)
