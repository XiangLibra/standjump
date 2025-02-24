from flask import Flask, request, jsonify, send_from_directory, render_template, abort
import shutil
import os
from pathlib import Path
import uuid
import cv2
import joblib
import pandas as pd
import re
from flask import Response
import subprocess
# 假設您有自定義的處理函數
from main import process_video_and_predict

app = Flask(__name__)

# 設置目錄
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

# 確保目錄存在
for directory in [RESULTS_DIR, UPLOAD_DIR]:
    directory.mkdir(exist_ok=True)

# 載入模型（假設您有預訓練模型）
model_path = BASE_DIR / "trained_model.joblib"
if not model_path.exists():
    raise FileNotFoundError("找不到 trained_model.joblib，請確保模型檔案存在。")
model = joblib.load(str(model_path))
print("✅ 模型已成功載入！")

# def images_to_video(image_folder, output_path, fps=2):
#     images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
#     if not images:
#         raise ValueError(f"找不到圖片檔案於資料夾：{image_folder}")
    
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, _ = frame.shape
    
#     # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 格式
#     video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     for image_name in images:
#         frame = cv2.imread(os.path.join(image_folder, image_name))
#         video.write(frame)
    
#     video.release()
def images_to_video(image_folder, output_path, fps=2):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        raise ValueError(f"❌ 找不到圖片檔案於資料夾：{image_folder}")

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # 使用 mp4v 產生 .avi 檔，再轉成 H.264
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_avi = output_path.replace('.mp4', '_temp.avi')

    video = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

    for image_name in images:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame)

    video.release()

    # 使用 FFmpeg 轉為 H.264 格式
    convert_to_h264(temp_avi, output_path)
    os.remove(temp_avi)

    print(f"✅ 影片成功生成：{output_path}")

def convert_to_h264(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-crf", "22",
        "-pix_fmt", "yuv420p", output_path
    ]
    subprocess.run(command, check=True)
    print(f"🎉 已轉為 H.264 MP4：{output_path}")
@app.route("/", methods=["GET"])
def read_root():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "未上傳影片檔案"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"status": "error", "message": "未選擇檔案"}), 400

    # 生成唯一的 process_id
    process_id = str(uuid.uuid4())
    print(f"Generated process_id: {process_id}")

    # 解析上傳檔名，準備做為輸出時的檔名基底
    original_filename = video.filename  # e.g. "myvideo.mp4" or "sample.MOV"
    base_name, ext = os.path.splitext(original_filename)  
    if not ext:
        ext = ".mp4"  # 若沒有副檔名，就預設 .mp4
    elif ext == ".mov":
        ext = ".mp4"
    elif ext == ".MOV":
        ext = ".mp4"
    elif ext == ".webm":
        ext = ".mp4"
    # 新的影片檔名(同使用者原檔名) e.g. "myvideo.mp4"
    processed_video_filename = base_name + ext
    # 新的 CSV 檔名
    output_csv_filename = base_name + ".csv"

    # 建立資料夾
    process_dir = UPLOAD_DIR / process_id
    result_dir = RESULTS_DIR / process_id
    process_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 儲存上傳的原始影片
    original_video_path = process_dir / original_filename
    video.save(str(original_video_path))

    try:
        # 產生預測用的暫存資料夾 /poses_images_out
        poses_out_dir = result_dir / "poses_images_out"
        poses_out_dir.mkdir(exist_ok=True)

        # CSV 輸出路徑
        csv_out_path = result_dir / output_csv_filename

        # 呼叫影片處理和預測
        predicted_value = process_video_and_predict(
            video_path=str(original_video_path),
            model=model,
            temp_folder=str(poses_out_dir),
            output_csvs_path=str(csv_out_path),
            fps_extract=2,
        )

        # 將處理後的影格合成影片
        processed_video_path = result_dir / processed_video_filename
        if poses_out_dir.exists():
            images_to_video(str(poses_out_dir), str(processed_video_path))

        # 檢查檔案是否存在
        if not processed_video_path.exists() or not csv_out_path.exists():
            print(f"錯誤：處理後檔案未生成，process_id: {process_id}")
            return jsonify({"status": "error", "message": "處理後檔案生成失敗"}), 500

        # 在 CSV 中新增欄位 "Predict length" 並寫入預測值
        try:
            df = pd.read_csv(csv_out_path)
            df['Predict length'] = predicted_value
            df.to_csv(csv_out_path, index=False)
        except Exception as csv_err:
            print(f"修改 CSV 時出錯: {csv_err}")

        # 若需要刪除上傳暫存資料夾
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)
            print(f"✅ 已自動刪除暫存資料夾: {process_dir}")

        # 回傳結果
        response_data = {
            "status": "success",
            "predicted_distance": float(predicted_value),
            "process_id": process_id,
            "video_filename": processed_video_filename,  # 改為使用者原檔名
            "csv_filename": output_csv_filename          # 對應 csv
        }
        print(f"回傳結果：{response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"發生錯誤：{str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====== 歷史紀錄相關 ======
@app.route("/api/history", methods=["GET"])
def get_history():
    if not RESULTS_DIR.exists():
        return jsonify({"folders": []})
    
    folders_info = []
    for folder in RESULTS_DIR.iterdir():
        if folder.is_dir():
            process_id = folder.name
            files = [f.name for f in folder.iterdir() if f.is_file()]
            folders_info.append({
                "process_id": process_id,
                "files": files
            })
    return jsonify({"folders": folders_info})

@app.route("/api/delete_folder/<process_id>", methods=["DELETE"])
def delete_folder(process_id):
    folder_path = RESULTS_DIR / process_id
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"status": "error", "message": "資料夾不存在"}), 404

    shutil.rmtree(folder_path)
    return jsonify({"status": "success", "message": f"資料夾 {process_id} 已刪除"})

@app.route("/api/delete_file", methods=["DELETE"])
def delete_file():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "缺少參數"}), 400

    process_id = data.get("process_id")
    filename = data.get("filename")

    if not process_id or not filename:
        return jsonify({"status": "error", "message": "參數不完整"}), 400

    file_path = RESULTS_DIR / process_id / filename
    if not file_path.exists():
        return jsonify({"status": "error", "message": "檔案不存在"}), 404

    file_path.unlink()
    return jsonify({"status": "success", "message": f"檔案 {filename} 已刪除"})

# ====== 一鍵刪除全部路由 ======
@app.route("/api/delete_all", methods=["DELETE"])
def delete_all():
    # 刪除 results 資料夾下所有子資料夾
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    return jsonify({"status": "success", "message": "已刪除全部資料夾"})

# ====== 下載或串流影片路由 ======
@app.route("/result/<process_id>/<filename>")
def download_file(process_id, filename):
    file_path = RESULTS_DIR / process_id / filename
    if not file_path.exists():
        return abort(404, description="檔案未找到")
    return send_from_directory(str(RESULTS_DIR / process_id), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
