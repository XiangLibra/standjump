import cv2
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import sys
import tqdm
import matplotlib
matplotlib.use('Agg')  # 避免 GUI 問題
import matplotlib.pyplot as plt

# 這些依賴檔案來自您先前的 Movenet 相關程式
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
import utils
from data import BodyPart
from ml import Movenet

# -----------------------
# 初始化 MoveNet (thunder 版本)
# -----------------------
movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
    """呼叫 MoveNet 進行單人姿態偵測."""
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
    """
    在圖像上畫出關節點。只回傳彩色圖像 (numpy)。
    """
    image_np = utils.visualize(image, [person])
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    ax.imshow(image_np)
    if close_figure:
        plt.close(fig)
    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))
    return image_np

# -----------------------
# MoveNet Preprocessor
# -----------------------
class MoveNetPreprocessor(object):
    """簡化版本的 MoveNet 前處理器。"""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_path):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

        os.makedirs(self._images_in_folder, exist_ok=True)
        os.makedirs(self._images_out_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self._csvs_out_path), exist_ok=True)

    def process(self, detection_threshold=0.1):
        """對資料夾內的影格執行 Movenet 偵測，並將結果儲存成 CSV 與標註後影像。"""
        print("🚀 開始處理圖片...")

        image_names = sorted([
            f for f in os.listdir(self._images_in_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')
        ])
        if not image_names:
            raise RuntimeError("❌ 沒有任何可用的影格。")

        with open(self._csvs_out_path, 'w', newline='') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # 寫入 CSV 表頭：file_name + 每個 keypoint (x, y, score)
            header = ["file_name"]
            for bodypart in BodyPart:
                header += [f"{bodypart.name}_x", f"{bodypart.name}_y", f"{bodypart.name}_score"]
            csv_out_writer.writerow(header)

            valid_image_count = 0

            for image_name in tqdm.tqdm(image_names):
                image_path = os.path.join(self._images_in_folder, image_name)
                try:
                    image_data = tf.io.read_file(image_path)
                    image = tf.io.decode_jpeg(image_data)
                except Exception as e:
                    self._messages.append(f"❌ Skipped {image_path}. Error: {e}")
                    continue

                # 檢查是否為 RGB
                if image.shape[-1] != 3:
                    self._messages.append(f"⚠️ Skipped {image_path}. Not RGB.")
                    continue

                # 呼叫 detect()
                person = detect(image)

                # 檢查關節點信心分數
                min_score = min([kp.score for kp in person.keypoints])
                if min_score < detection_threshold:
                    self._messages.append(f"⚠️ Skipped {image_path}. Low confidence.")
                    continue

                valid_image_count += 1
                # 繪製姿態
                output_overlay = draw_prediction_on_image(
                    image.numpy().astype(np.uint8),
                    person,
                    close_figure=True,
                    keep_input_size=True
                )
                output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
                output_image_path = os.path.join(self._images_out_folder, image_name)
                cv2.imwrite(output_image_path, output_frame)

                # 寫入 CSV
                pose_landmarks = np.array(
                    [[kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints],
                    dtype=np.float32
                )
                coordinates = pose_landmarks.flatten().astype(str).tolist()
                csv_out_writer.writerow([image_name] + coordinates)

        if valid_image_count == 0:
            raise RuntimeError("❌ 沒有找到有效的圖片。")
        else:
            print(f"✅ 完成：共處理 {valid_image_count} 張影格。")

        if self._messages:
            print("\n".join(self._messages))


# -----------------------
# 新增的預測函式 (含 height, weight)
# -----------------------
def predict_from_csv(csv_path, model, scaler, height, weight):
    """
    從 MoveNet 產出的 CSV 讀取特徵，並手動加入 height & weight 進行預測。
    """
    df = pd.read_csv(csv_path)
    if "file_name" in df.columns:
        df.drop("file_name", axis=1, inplace=True)
    # 缺失值填0
    df.fillna(0, inplace=True)

    # 這裡範例：取多行 (多張影格) 的「最大值」作為特徵 (可依需求更改)
    feature_vector = df.max(axis=0).values.reshape(1, -1)

    # 把身高 & 體重做為新特徵附加到最後
    feature_vector = np.append(feature_vector, [height, weight]).reshape(1, -1)

    # 避免超出模型訓練範圍：可做 clip，也可自由調整
    # 這裡假設我們用 scaler.mean_ 和 scaler.var_ 來 roughly clip
    feature_vector = np.clip(
        feature_vector,
        scaler.mean_ - 3*np.sqrt(scaler.var_),
        scaler.mean_ + 3*np.sqrt(scaler.var_)
    )

    # 標準化
    feature_vector_scaled = scaler.transform(feature_vector)

    # 預測
    prediction = model.predict(feature_vector_scaled)[0]
    # 避免出現負數
    prediction = max(0, prediction)

    return prediction


# -----------------------
# 整合主流程
# -----------------------
def process_video_and_predict(
    video_path,
    model,
    scaler,
    temp_folder='temp_frames',
    output_csvs_path='output_csvs/output_data.csv',
    fps_extract=2,
    height=148.0,     # 新增參數，預設148
    weight=38.0       # 新增參數，預設38
):
    """
    1) 先把影片擷取成影格存在 temp_folder
    2) 用 MoveNet 偵測影格、輸出 CSV
    3) 依照 height, weight 做最終預測
    4) 回傳 predicted_value
    """
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # 1. 轉影片為影格
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30  # 預設30fps
    frame_interval = int(video_fps / fps_extract) if fps_extract > 0 else 1

    frame_count, extracted_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"{extracted_count:06d}.jpg"
            cv2.imwrite(os.path.join(temp_folder, frame_name), frame)
            extracted_count += 1

        frame_count += 1
    cap.release()
    print(f"已擷取 {extracted_count} 張影格至: {temp_folder}")

    # 2. 使用 MoveNet 處理影格並輸出 CSV
    preprocessor = MoveNetPreprocessor(
        images_in_folder=temp_folder,
        images_out_folder=temp_folder,  # 偵測後圖片也放在同一個資料夾
        csvs_out_path=str(output_csvs_path)
    )
    print("🚀 開始偵測姿態並輸出 CSV ...")
    preprocessor.process()

    # 3. 使用模型 & scaler 進行預測 (帶入身高體重)
    predicted_value = predict_from_csv(
        csv_path=output_csvs_path,
        model=model,
        scaler=scaler,
        height=height,
        weight=weight
    )
    print(f"影片最終預測結果: {predicted_value}")
    return predicted_value
