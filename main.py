import cv2
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 後端避免 GUI 問題
import matplotlib.pyplot as plt
import shutil

import csv
import sys
import numpy as np
import tensorflow as tf
import tqdm

pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
import utils
from data import BodyPart
from ml import Movenet

movenet = Movenet('movenet_thunder')
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.
 
  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.
 
  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape
 
  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)
 
  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  """Draws the keypoint predictions on image.
 
  Args:
    image: An numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.
 
  Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])
  
  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)
 
  if close_figure:
    plt.close(fig)
 
  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np
class MoveNetPreprocessor(object):
    """Simplified MoveNet Preprocessor without subfolders."""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_path):
        """
        Args:
            images_in_folder: 存放輸入圖片的資料夾。
            images_out_folder: 存放偵測後圖片的資料夾。
            csvs_out_path: 存放關節點資料的 CSV 檔案路徑。
        """
        if not os.path.exists(images_in_folder):
            os.makedirs(images_in_folder)

        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

        os.makedirs(self._images_out_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self._csvs_out_path), exist_ok=True)

        # 取得所有圖片檔案名稱 (.jpg, .png)


    def process(self, detection_threshold=0.1):
        """處理所有圖片，進行姿勢偵測並輸出結果到圖片和 CSV。"""
        print("🚀 開始處理圖片...")

        self._image_names = sorted([
            f for f in os.listdir(self._images_in_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')
        ])
        # 建立 CSV 檔案，寫入表頭
        with open(self._csvs_out_path, 'w', newline='') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            # 表頭：file_name + 每個 keypoint 的 (x, y, score)
            header = ["file_name"]
            for bodypart in BodyPart:
                header += [f"{bodypart.name}_x", f"{bodypart.name}_y", f"{bodypart.name}_score"]
            csv_out_writer.writerow(header)

            valid_image_count = 0

            # 逐張圖片進行姿勢偵測
            for image_name in tqdm.tqdm(self._image_names):
                image_path = os.path.join(self._images_in_folder, image_name)

                # 嘗試讀取圖片
                try:
                    image = tf.io.read_file(image_path)
                    image = tf.io.decode_jpeg(image)
                except Exception as e:
                    self._messages.append(f"❌ Skipped {image_path}. Error: {e}")
                    continue

                # 檢查是否為 RGB 圖片
                if image.shape[-1] != 3:
                    self._messages.append(f"⚠️ Skipped {image_path}. Not RGB.")
                    continue

                # 呼叫偵測函數
                person = detect(image)

                # 檢查信心分數
                min_landmark_score = min([k.score for k in person.keypoints])
                if min_landmark_score < detection_threshold:
                    self._messages.append(f"⚠️ Skipped {image_path}. Low confidence.")
                    continue

                valid_image_count += 1

                # 繪製關節點
                output_overlay = draw_prediction_on_image(
                    image.numpy().astype(np.uint8),
                    person,
                    close_figure=True,
                    keep_input_size=True
                )
                output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)

                # 儲存繪製後的圖片
                # self._images_out_folder
                output_image_path = os.path.join(self._images_out_folder, image_name)
#                print(f"✅ 已儲存繪製後的圖片: {self._images_out_folder}")
                cv2.imwrite(output_image_path, output_frame)
               

                # 取出 keypoint 座標與分數，並寫入 CSV
                pose_landmarks = np.array(
                    [[kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints],
                    dtype=np.float32
                )
                coordinates = pose_landmarks.flatten().astype(str).tolist()
                csv_out_writer.writerow([image_name] + coordinates)
            # 4️⃣ 自動刪除 temp_frames
        # if os.path.exists(self._images_in_folder):
        #     shutil.rmtree(self._images_in_folder)
        #     print(f"✅ 已自動刪除暫存資料夾: {self._images_in_folder}")

        # 結束訊息
        if valid_image_count == 0:
            raise RuntimeError("❌ 沒有找到有效的圖片進行姿勢偵測。")
        else:
            print(f"✅ 處理完成，共處理 {valid_image_count} 張有效圖片。")

        if self._messages:
            print("\n".join(self._messages))



# 假設有影片的 csv 檔案 (經過 MoveNet 處理後)
def predict_from_csv(csv_path, model):
    import pandas as pd

    # 讀取 CSV 並處理資料
    df = pd.read_csv(csv_path)
    
    if "file_name" in df.columns:
        df.drop("file_name", axis=1, inplace=True)

    # 聚合多行數據 (取平均)
    feature_vector = df.min(axis=0).values.reshape(1, -1)
    # row_series = df_temp.max(axis=0)

    # 預測
    prediction = model.predict(feature_vector)
    return prediction[0]
def process_video_and_predict(video_path, model, temp_folder='temp_frames',output_csvs_path='output_csvs/output_data.csv', fps_extract=2):
    # 1. 轉影片為影格
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_extract)
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
    print(f"已擷取 {extracted_count} 張影格")

    # 2. 使用 MoveNet 處理影格並輸出 CSV
    # 假設此步驟已經完成，並生成了 `movenet_output.csv`
    preprocessor = MoveNetPreprocessor(
    # images_in_folder='temp_frames',        # 輸入圖片資料夾
    images_in_folder=temp_folder,        # 輸入圖片資料夾
    # images_out_folder='poses_images_out',   # 偵測後圖片輸出
    images_out_folder=temp_folder,   # 偵測後圖片輸出
    
    # csvs_out_path='output_csvs/output_data.csv'  # CSV 輸出
     csvs_out_path=str(output_csvs_path) # CSV 輸
        )
    print("🚀 開始執行 MoveNetPreprocessor.process() ...")
    preprocessor.process()

    # 3. 使用已訓練好的模型進行預測
    csv_output_path = output_csvs_path  # 假設 MoveNet 的結果
    predicted_value = predict_from_csv(csv_output_path, model)
    print(f"影片預測結果: {predicted_value}")
    return predicted_value

#讀取影片
# video_path = "CAO,RUEI-LIAN(1).MOV"
video_path = "YANG,YI-RU(4).MOV"  

preprocessor = MoveNetPreprocessor(
    images_in_folder='temp_frames',        # 輸入圖片資料夾
    images_out_folder='poses_images_out',   # 偵測後圖片輸出
    csvs_out_path='output_csvs/output_data.csv'  # CSV 輸出
)

import joblib

# 假設已訓練好的 model
model = LinearRegression()


# 儲存模型
model_save_path = "trained_model.joblib"

# 載入儲存的模型
loaded_model = joblib.load(model_save_path)
print("模型已成功載入")
#lx=process_video_and_predict(video_path, loaded_model)


