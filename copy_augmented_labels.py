import os

def copy_labels_for_augmented_images(original_base_name, output_base_name, input_labels_folder, output_txt_folder, num_augmentations=20):
    original_txt_path = os.path.join(input_labels_folder, f"{original_base_name}.txt")

    # Sprawdź, czy plik etykiet istnieje
    if os.path.exists(original_txt_path):
        with open(original_txt_path, 'r') as original_file:
            content = original_file.read()

        for i in range(1, num_augmentations + 1):
            new_base_name = f"{output_base_name}_{i}"
            new_txt_path = os.path.join(output_txt_folder, f"{new_base_name}.txt")

            with open(new_txt_path, 'w') as new_file:
                new_file.write(content)

def main():
    input_labels_folder = r"C:\Users\mateu\Desktop\wykrywanie_rzeczy\dataset\labels"
    output_txt_folder = r"C:\Users\mateu\Desktop\wykrywanie_rzeczy\modyfikacja_txt"

    if not os.path.exists(output_txt_folder):
        os.makedirs(output_txt_folder)

    for filename in os.listdir(input_labels_folder):
        if filename.endswith(".txt"):
            original_base_name = os.path.splitext(filename)[0]
            copy_labels_for_augmented_images(original_base_name, original_base_name, input_labels_folder, output_txt_folder)

if __name__ == "__main__":
    main()
