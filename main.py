# main.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用所有 GPU 裝置
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 避免 GUI 問題
import matplotlib.pyplot as plt
import joblib

# =============== Movenet 相關 ===============
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
import utils
from data import BodyPart
from ml import Movenet

# 初始化 MoveNet (thunder 版本)
movenet = Movenet('movenet_thunder')

def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
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

class MoveNetPreprocessor:
    """將某資料夾裡的影格做 MoveNet 偵測，輸出 CSV."""
    def __init__(self, images_in_folder, images_out_folder, csvs_out_path):
        os.makedirs(images_in_folder, exist_ok=True)
        os.makedirs(images_out_folder, exist_ok=True)
        os.makedirs(os.path.dirname(csvs_out_path), exist_ok=True)
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

    def process(self, detection_threshold=0.1):
        print("🚀 MoveNet Preprocessor: 開始處理圖片...")
        image_names = sorted([
            f for f in os.listdir(self._images_in_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')
        ])
        if not image_names:
            raise RuntimeError("❌ 沒有任何可用的影格。")

        with open(self._csvs_out_path, 'w', newline='') as csv_out_file:
            import csv
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # CSV 表頭
            header = ["file_name"]
            for bodypart in BodyPart:
                header += [f"{bodypart.name}_x", f"{bodypart.name}_y", f"{bodypart.name}_score"]
            csv_out_writer.writerow(header)

            valid_count = 0

            for image_name in tqdm.tqdm(image_names):
                image_path = os.path.join(self._images_in_folder, image_name)
                try:
                    image_data = tf.io.read_file(image_path)
                    image = tf.io.decode_jpeg(image_data)
                except Exception as e:
                    self._messages.append(f"❌ Skipped {image_path}, Error: {e}")
                    continue

                if image.shape[-1] != 3:
                    self._messages.append(f"⚠️ Skipped {image_path}. Not RGB.")
                    continue

                # MoveNet 偵測
                person = detect(image)
                min_score = min([kp.score for kp in person.keypoints])
                if min_score < detection_threshold:
                    self._messages.append(f"⚠️ Skipped {image_path}. Low confidence.")
                    continue

                valid_count += 1
                # 畫關節
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
                pose_landmarks = np.array([
                    [kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints
                ], dtype=np.float32)
                row_data = pose_landmarks.flatten().astype(str).tolist()
                csv_out_writer.writerow([image_name] + row_data)

        if valid_count == 0:
            raise RuntimeError("❌ 沒有找到有效圖片。")
        else:
            print(f"✅ MoveNet: 完成處理 {valid_count} 張影格。")

        if self._messages:
            print("\n".join(self._messages))

# =============== 新增的「特徵計算 + RMS + 預測」程式 ===============
# 請把您提供的程式碼放這裡 (稍做整合)

BODY_PARTS = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

def compute_rms_features(df_temp):
    features_dict = {}
    for part in BODY_PARTS:
        x_col = f"{part}_x"
        y_col = f"{part}_y"
        s_col = f"{part}_score"
        if x_col not in df_temp.columns or y_col not in df_temp.columns or s_col not in df_temp.columns:
            continue

        # (A) 加權平均
        x_score_series = df_temp[x_col] * df_temp[s_col]
        y_score_series = df_temp[y_col] * df_temp[s_col]
        features_dict[f"{part}_x_score_mean"] = x_score_series.mean()
        features_dict[f"{part}_y_score_mean"] = y_score_series.mean()

        # (B) 速度衍生 RMS
        dx = df_temp[x_col].diff().fillna(0)
        dy = df_temp[y_col].diff().fillna(0)
        speed = np.sqrt(dx**2 + dy**2)

        features_dict[f"RMS_{part}_max"] = speed.max()
        features_dict[f"RMS_{part}_min"] = speed.min()
        features_dict[f"RMS_{part}_mean"] = speed.mean()
        features_dict[f"RMS_{part}_std"] = speed.std()

        speed_shifted = speed.shift(1)
        valid_mask = (speed_shifted != 0)
        if valid_mask.sum() > 1:
            speed_diff_ratio = (speed - speed_shifted) / (speed_shifted + 1e-8)
            roc_val = speed_diff_ratio[valid_mask].mean()
        else:
            roc_val = 0.0
        features_dict[f"RMS_{part}_roc"] = roc_val
    return features_dict

def read_time_series_csv_as_one_row(csv_path, agg_mode="mean"):
    df_temp = pd.read_csv(csv_path)
    if len(df_temp) == 0:
        return pd.Series(dtype=float)

    if "file_name" in df_temp.columns:
        df_temp.drop("file_name", axis=1, inplace=True)

    # 預設平均，也可 max, min, last
    if agg_mode == "mean":
        row_series = df_temp.mean(axis=0)
    elif agg_mode == "max":
        row_series = df_temp.max(axis=0)
    elif agg_mode == "min":
        row_series = df_temp.min(axis=0)
    elif agg_mode == "last":
        row_series = df_temp.iloc[-1]
    else:
        row_series = df_temp.mean(axis=0)

    # RMS 特徵
    rms_dict = compute_rms_features(df_temp)
    for k, v in rms_dict.items():
        row_series[k] = v

    return row_series

def read_csv(csv_path, height, weight, agg_mode="mean"):
    row_series = read_time_series_csv_as_one_row(csv_path, agg_mode)
    # 加入身高體重
    row_series["height"] = height
    row_series["weight"] = weight
    return row_series

def ensure_features_match(row_series, expected_features):
    """
    缺少的欄位補0；並確保順序與模型訓練時相同。
    """
    missing_features = set(expected_features) - set(row_series.index)
    for feat in missing_features:
        row_series[feat] = 0
    return row_series[expected_features]

def predict_from_csv(csv_path, model, scaler, selected_features, height, weight):
    row_series = read_csv(csv_path, height, weight, agg_mode="mean")
    row_series = ensure_features_match(row_series, selected_features)
    feature_vector = scaler.transform([row_series.values])
    prediction = model.predict(feature_vector)[0]
    return max(0, prediction)

# =============== 整合 MoveNet + 新預測流程 ===============
def process_video_and_predict(
    video_path,           # 影片路徑
    model,                # 已載入之模型
    scaler,               # 已載入之 scaler
    selected_features,    # 模型訓練時用到的特徵清單
    temp_folder='temp_frames',
    csv_out_path='output_csvs/output_data.csv',
    fps_extract=2,
    height=148.0,
    weight=38.0
):
    """
    1) 擷取影片 → temp_folder
    2) MoveNet 偵測 → 產 CSV
    3) 使用 predict_from_csv(...) 計算 RMS 特徵 + 加入身高體重 → 預測
    4) 回傳 predicted_value
    """
    # 建立資料夾
    os.makedirs(temp_folder, exist_ok=True)

    # 1) 擷取影片影格
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
    interval = int(fps / fps_extract) if fps_extract > 0 else 1

    frame_cnt, extracted = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt % interval == 0:
            img_name = f"{extracted:06d}.jpg"
            cv2.imwrite(os.path.join(temp_folder, img_name), frame)
            extracted += 1
        frame_cnt += 1
    cap.release()
    print(f"影片擷取完成: {extracted} 張影格 -> {temp_folder}")

    # 2) MoveNet 偵測 & 產 CSV
    preprocessor = MoveNetPreprocessor(
        images_in_folder=temp_folder,
        images_out_folder=temp_folder,  # 偵測後圖片仍放在同資料夾
        csvs_out_path=csv_out_path
    )
    preprocessor.process()

    # 3) 執行您新增的「RMS + 身高體重」預測
    predicted_value = predict_from_csv(
        csv_path=csv_out_path,
        model=model,
        scaler=scaler,
        selected_features=selected_features,
        height=height,
        weight=weight
    )
    print(f"最終預測距離: {predicted_value}")
    return predicted_value