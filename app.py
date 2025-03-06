from flask import Flask, request, jsonify, send_from_directory, render_template, abort
import shutil
import os
from pathlib import Path
import uuid
import cv2
import joblib
import pandas as pd
from flask import Response

# 載入您寫好的函式
from main import process_video_and_predict

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

# 確保目錄存在
for directory in [RESULTS_DIR, UPLOAD_DIR]:
    directory.mkdir(exist_ok=True)

# 1) 載入模型 & scaler（假設名為 best_trained_model.joblib）
model_path = BASE_DIR / "best_trained_model.joblib"
if not model_path.exists():
    raise FileNotFoundError("找不到 best_trained_model.joblib，請確保模型檔案存在。")
loaded_model, scaler = joblib.load(str(model_path))
print("✅ (app.py) 模型與 scaler 已成功載入！")


def images_to_video(image_folder, output_path, fps=2):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        raise ValueError(f"找不到圖片檔案於資料夾：{image_folder}")
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for image_name in images:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        video.write(frame)
    video.release()


@app.route("/", methods=["GET"])
def index():
    """主頁，渲染您的 index.html"""
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    接收影片、身高、體重，執行姿態偵測 & 預測，最後回傳 JSON。
    """
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "未上傳影片檔案"}), 400

    video = request.files['video']
    if not video or video.filename == '':
        return jsonify({"status": "error", "message": "未選擇檔案"}), 400

    # 取得身高與體重 (若前端沒帶，就預設148 / 38)
    height_str = request.form.get('height', '')
    weight_str = request.form.get('weight', '')
    try:
        height_val = float(height_str) if height_str else 148.0
    except:
        height_val = 148.0
    try:
        weight_val = float(weight_str) if weight_str else 38.0
    except:
        weight_val = 38.0

    # 產生 process_id
    process_id = str(uuid.uuid4())
    print(f"Generated process_id: {process_id}")

    # 拆解副檔名
    original_filename = video.filename
    base_name, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".mp4"
    elif ext.lower() in [".mov", ".webm"]:
        ext = ".mp4"

    # 準備路徑
    process_dir = UPLOAD_DIR / process_id
    result_dir = RESULTS_DIR / process_id
    process_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 儲存上傳檔案
    saved_video_path = process_dir / f"{base_name}{ext}"
    video.save(str(saved_video_path))

    # 決定輸出檔名
    processed_video_filename = f"{base_name}{ext}"
    output_csv_filename = f"{base_name}.csv"

    try:
        # Movenet 輸出 CSV 路徑
        csv_out_path = result_dir / output_csv_filename
        # 先處理影片 & 取得預測
        predicted_value = process_video_and_predict(
            video_path=str(saved_video_path),
            model=loaded_model,
            scaler=scaler,
            temp_folder=str(result_dir / "poses_images_out"),  # 影格和標記後圖都放一起
            output_csvs_path=str(csv_out_path),
            fps_extract=2,
            height=height_val,   # 傳入身高
            weight=weight_val    # 傳入體重
        )

        # 合成影片
        processed_video_path = result_dir / processed_video_filename
        if os.path.exists(str(result_dir / "poses_images_out")):
            images_to_video(str(result_dir / "poses_images_out"), processed_video_path, fps=2)

        if not processed_video_path.exists() or not csv_out_path.exists():
            return jsonify({"status": "error", "message": "處理後檔案生成失敗"}), 500

        # 在 CSV 中加入 "Predict length" 欄位
        try:
            df = pd.read_csv(csv_out_path)
            df['Predict length'] = predicted_value
            df.to_csv(csv_out_path, index=False)
        except Exception as csv_err:
            print(f"修改 CSV 時出錯: {csv_err}")

        # (選擇) 刪除上傳暫存資料夾
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)
            print(f"✅ 已自動刪除暫存上傳資料夾: {process_dir}")

        response_data = {
            "status": "success",
            "predicted_distance": float(predicted_value),
            "process_id": process_id,
            "video_filename": processed_video_filename,
            "csv_filename": output_csv_filename
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"發生錯誤：{e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============= 以下為您原先的歷史紀錄 / 刪除 等 API =============

@app.route("/api/history", methods=["GET"])
def get_history():
    if not RESULTS_DIR.exists():
        return jsonify({"folders": []})
    folders_info = []
    for folder in RESULTS_DIR.iterdir():
        if folder.is_dir():
            pid = folder.name
            files = [f.name for f in folder.iterdir() if f.is_file()]
            folders_info.append({
                "process_id": pid,
                "files": files
            })
    return jsonify({"folders": folders_info})

@app.route("/api/delete_folder/<process_id>", methods=["DELETE"])
def delete_folder(process_id):
    folder_path = RESULTS_DIR / process_id
    if not folder_path.exists():
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

@app.route("/api/delete_all", methods=["DELETE"])
def delete_all():
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    return jsonify({"status": "success", "message": "已刪除全部資料夾"})

@app.route("/result/<process_id>/<filename>")
def download_file(process_id, filename):
    file_path = RESULTS_DIR / process_id / filename
    if not file_path.exists():
        return abort(404, description="檔案未找到")
    return send_from_directory(str(RESULTS_DIR / process_id), filename, as_attachment=True)

if __name__ == "__main__":
    # 指定 port=5000
    app.run(host="127.0.0.1", port=5000, debug=True)
