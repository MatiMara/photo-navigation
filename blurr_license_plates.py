import pandas
import torch
from time import time
import os
import cv2
import numpy as np
from podzial import *
import warnings


def anonymize_license_plate(image, factor=3.0):
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)  #filtr Gausowski


def anonymize_license_plate_pixelate(image, blocks=3):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    return image


def blurr_images(license_model, input_path="../test", output_path="results", batch_size=1):
    images = [f"{input_path}/{f}" for f in os.listdir(input_path)]

    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir(output_path):
        results_dir_num += 1
    os.mkdir(f"{output_path}/exp{results_dir_num}")

    start = time()

    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]

        results = license_model(batch_images)

        for i, res_df in enumerate(results.pandas().xyxy):
            img_idx = start_idx + i
            img_path = images[img_idx]
            img = cv2.imread(img_path)

            for j in range(len(res_df)):
                xmin = round(res_df['xmin'][j])
                ymin = round(res_df['ymin'][j])
                xmax = round(res_df['xmax'][j])
                ymax = round(res_df['ymax'][j])
                license_plate = img[ymin:ymax, xmin:xmax]
                img[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)

            output_folder = f"{output_path}/exp{results_dir_num}"
            output_filename = f"output{img_idx}.jpg"
            cv2.imwrite(os.path.join(output_folder, output_filename), img)

    end = time()
    timer = end - start
    print('Czas:', timer, 'sekund')


def blurr_video(model, input_path, output_path="results"): #działa w starszej wersji
    video = cv2.VideoCapture(input_path)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir("results"):
        results_dir_num += 1
    os.mkdir(f"{output_path}/exp{results_dir_num}")

    output = cv2.VideoWriter(f"{output_path}/exp{results_dir_num}/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    while True:
        ret, frame = video.read()
        if ret:
            result = model(frame)
            res_df = result.pandas().xyxy[0]
            for j in range(len(res_df)):
                if res_df['confidence'][j] < 0.2:
                    continue
                xmin = round(res_df['xmin'][j])
                ymin = round(res_df['ymin'][j])
                xmax = round(res_df['xmax'][j])
                ymax = round(res_df['ymax'][j])
                license_plate = frame[ymin:ymax, xmin:xmax]
                frame[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)   # change
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    cv2.destroyAllWindows()
    video.release()
    output.release()


def blurr_images_split(license_model, input_path="../test", output_path="results"):
    images = [f"{input_path}/{f}" for f in os.listdir(input_path)]  # or file, Path, PIL, OpenCV, numpy, list

    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir(output_path):
        results_dir_num += 1
    os.mkdir(f"{output_path}/exp{results_dir_num}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    start = time()

    results_after_combining = []
    for it, im in enumerate(images):
        result_after_combining = pandas.DataFrame(columns=["xmin", "ymin", "xmax", "ymax"])
        split_photo(im)
        split_images = [f"C:\\Users\\mateu\\Desktop\\t2\\{f}" for f in os.listdir("C:\\Users\\mateu\\Desktop\\t2")]
        # print(f"split_images={split_images}")
        split_results = license_model(split_images)
        # print(f"split_results={split_results}")
        for i, res in enumerate(split_results.pandas().xyxy):
            split_img_name = split_images[i].split("/")[-1]
            # print(len(split_img_name.split("_")), split_img_name.split("_"))
            if len(split_img_name.split("_")) > 1:
                start_x = int(split_img_name.split("_")[-3])
                start_y = int(split_img_name.split("_")[-2])
                # print(f"res={res}")
                for _, r in res[["xmin", "ymin", "xmax", "ymax"]].iterrows():
                    # print("r before adding = ", r)
                    r["xmin"] += start_x
                    r["ymin"] += start_y
                    r["xmax"] += start_x
                    r["ymax"] += start_y
                    # print("r = ", r)
                    result_after_combining = result_after_combining.append(r)
                    # print("result_after_combining = ", result_after_combining)
            else:
                for _, r in res[["xmin", "ymin", "xmax", "ymax"]].iterrows():
                    # print(type(res), res)
                    # print(type(res[["xmin", "ymin", "xmax", "ymax"]]), res[["xmin", "ymin", "xmax", "ymax"]])
                    # print("r = ", r)
                    result_after_combining = result_after_combining.append(r)
                    # print("result_after_combining = ", result_after_combining)

        results_after_combining.append(result_after_combining)
        # print(im.split("/")[-1])
        # print(result_after_combining)
        # print(f"{it+1}/{len(images)}")

        # Po ilu zdjęciach przerwać
        x = 1000
        if it+1 == x:
            break
    # print(results_after_combining)

    for i, res_df in enumerate(results_after_combining):
        res_df = res_df.reset_index(drop=True)
        # print(i)
        img = cv2.imread(images[i])
        for j in range(len(res_df)):
            # print(res_df)
            xmin = round(res_df['xmin'][j])
            ymin = round(res_df['ymin'][j])
            xmax = round(res_df['xmax'][j])
            ymax = round(res_df['ymax'][j])
            # print(ymin, ymax, xmin, xmax)
            license_plate = img[ymin:ymax, xmin:xmax]
            img[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)  # change
        cv2.imwrite(f"{output_path}/exp{results_dir_num}/output{i}.jpg", img)

    end = time()
    timer = end - start
    print('czas:', timer, 'sekund')


def blurr_images_vehicle(license_model, vehicle_model, input_path="../test", output_path="wyniki", batch_size=3):
    results_dir_num = 0
    while f"exp{results_dir_num}" in os.listdir(output_path):
        results_dir_num += 1
    exp_output_path = os.path.join(output_path, f"exp{results_dir_num}")
    os.makedirs(exp_output_path, exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    start = time()

    vehicle_model.conf = 0.1

    images = [f"{input_path}/{f}" for f in os.listdir(input_path)]
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx]

        vehicles_results = vehicle_model(batch_images)
        vehicles_results_ls = vehicles_results.pandas().xyxy

        results_after_combining = []
        for it, im in enumerate(batch_images):
            result_after_combining = pandas.DataFrame(columns=["xmin", "ymin", "xmax", "ymax"])
            vehicles_results_df = vehicles_results_ls[it]
            veh_res_df = vehicles_results_df[vehicles_results_df['class'].isin([2, 3, 5, 7])]
            veh_res_df = veh_res_df.reset_index()
            split_photo(im, veh_res_df)
            split_images = [f"C:\\Users\\mateu\\Desktop\\t2\\{f}" for f in os.listdir("C:\\Users\\mateu\\Desktop\\t2")]
            split_results = license_model(split_images)
            for i, res in enumerate(split_results.pandas().xyxy):
                split_img_name = split_images[i].split("/")[-1]
                if len(split_img_name.split("_")) > 1:
                    start_x = int(split_img_name.split("_")[-3])
                    start_y = int(split_img_name.split("_")[-2])
                    for _, r in res[["xmin", "ymin", "xmax", "ymax"]].iterrows():
                        r["xmin"] += start_x
                        r["ymin"] += start_y
                        r["xmax"] += start_x
                        r["ymax"] += start_y
                        result_after_combining = result_after_combining.append(r)
                else:
                    for _, r in res[["xmin", "ymin", "xmax", "ymax"]].iterrows():
                        result_after_combining = result_after_combining.append(r)

            results_after_combining.append(result_after_combining)
            #print(f"{it+1}/{len(batch_images)}")

        for i, res_df in enumerate(results_after_combining):
            res_df = res_df.reset_index(drop=True)
            img = cv2.imread(batch_images[i])
            for j in range(len(res_df)):
                xmin = round(res_df['xmin'][j])
                ymin = round(res_df['ymin'][j])
                xmax = round(res_df['xmax'][j])
                ymax = round(res_df['ymax'][j])
                license_plate = img[ymin:ymax, xmin:xmax]
                img[ymin:ymax, xmin:xmax] = anonymize_license_plate_pixelate(license_plate, 10)
            output_filename = f"output{i + batch_idx * batch_size}.jpg"
            output_filepath = os.path.join(exp_output_path, output_filename)
            cv2.imwrite(output_filepath, img)

        # Clear memory
        del vehicles_results, vehicles_results_ls, results_after_combining

    end = time()
    timer = end - start
    print('Czas:', timer, 'sekund')


if __name__ == "__main__":
    yolo_license_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\mateu\\Desktop\\bluring_plates\\yolov5\\runs\\train\\exp61\\weights\\best.pt')
    yolo_vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\mateu\\Desktop\\bluring_plates\\yolov5\\yolov5x6.pt')
    blurr_images_vehicle(license_model=yolo_license_model, vehicle_model=yolo_vehicle_model, input_path="C:\\Users\\mateu\\Desktop\\mag\\Rejestracje_testy", output_path="C:\\Users\\mateu\\Desktop\\mag\\wyniki", batch_size=3)
    # blurr_images_split(license_model=yolo_license_model, input_path="C:\\Users\\mateu\\Desktop\\mag\\Rejestracje_testy", output_path="C:\\Users\\mateu\\Desktop\\mag\\wyniki")
    # blurr_images(license_model=yolo_license_model, input_path="C:\\Users\\mateu\\Desktop\\mag\\Rejestracje_testy", output_path="C:\\Users\\mateu\\Desktop\\mag\\wyniki", batch_size=1)
    # blurr_video(model=yolo_model, input_path="test_videos/test3.mp4", output_path="results")
    # split_photo("11.jpg")
    # 2 etapy ze zwalnianiem pamięci ew. zmniejszyć bacza