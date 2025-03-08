# app.py
from flask import Flask, request, jsonify, send_from_directory, render_template, abort
import os
import shutil
import uuid
from pathlib import Path
import cv2
import pandas as pd
import joblib
from flask import Response
import time

# 匯入我們在 main.py 定義的處理函式
from main import process_video_and_predict

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
for d in [UPLOAD_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ====== 載入模型 & scaler & 特徵清單 ======
best_model_path = BASE_DIR / "best_model.joblib"
if not best_model_path.exists():
    raise FileNotFoundError("找不到 best_model.joblib")

loaded_model, scaler, TRAIN_FEATURES = joblib.load(str(best_model_path))
print("✅ 已載入 best_model.joblib, 取得 model, scaler 以及訓練特徵列表。")

def images_to_video(image_folder, output_path, fps=2):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        raise ValueError(f"沒有任何影格 JPG 檔於: {image_folder}")

    frame0 = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame0.shape
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # h264
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        out.write(frame)
    out.release()

@app.route("/", methods=["GET"])
def index():
    # 回傳您的前端 index.html
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    上傳影片 + height + weight → Movenet + 預測 → 回傳 JSON。
    """
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "沒有收到影片"}), 400

    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"status": "error", "message": "未選擇檔案"}), 400

    # 從前端 FormData 取得身高、體重
    # 若沒填，就預設 148/38
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

    process_id = str(uuid.uuid4())
    print(f"[analyze_video] process_id = {process_id}")

    # 建立對應資料夾
    process_dir = UPLOAD_DIR / process_id
    result_dir = RESULTS_DIR / process_id
    process_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 儲存上傳影片
    original_filename = file.filename
    base_name, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".mp4"
    elif ext.lower() in [".mov", ".webm"]:
        ext = ".mp4"
    saved_video_path = process_dir / (base_name + ext)
    file.save(str(saved_video_path))

    # 產生輸出檔名
    output_csv_filename = base_name + ".csv"
    processed_video_filename = base_name + ext

    # CSV 路徑
    csv_out_path = result_dir / output_csv_filename
    # Movenet 產生標註影格的資料夾(也可直接放 result_dir)
    poses_folder = result_dir / "poses_images_out"
    poses_folder.mkdir(exist_ok=True)

    try:
        # 執行 Movenet + RMS 預測
        predicted_value = process_video_and_predict(
            video_path=str(saved_video_path),
            model=loaded_model,
            scaler=scaler,
            selected_features=TRAIN_FEATURES,
            temp_folder=str(poses_folder),
            csv_out_path=str(csv_out_path),
            fps_extract=2,
            height=height_val,
            weight=weight_val
        )

        # 合成標註後影片
        processed_video_path = result_dir / processed_video_filename
        images_to_video(str(poses_folder), processed_video_path, fps=2)

        # 確認產物
        if not processed_video_path.exists() or not csv_out_path.exists():
            return jsonify({"status": "error", "message": "處理後檔案生成失敗"}), 500

        # 在 CSV 加上欄位 "Predict length"
        try:
            df = pd.read_csv(csv_out_path)
            df['Predict length'] = predicted_value
            df.to_csv(csv_out_path, index=False)
        except Exception as csv_err:
            print("在 CSV 加欄位時出錯：", csv_err)

        # (選擇性) 清理上傳暫存檔
        shutil.rmtree(process_dir, ignore_errors=True)

        resp = {
            "status": "success",
            "predicted_distance": float(predicted_value),
            "process_id": process_id,
            "video_filename": processed_video_filename,
            "csv_filename": output_csv_filename
        }
        return jsonify(resp)

    except Exception as e:
        print("analyze_video 發生錯誤：", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# 其餘歷史紀錄、刪除檔案等路由...
# (省略，您可依原本程式保留)

# @app.route("/api/history", methods=["GET"])
# def get_history():
#     if not RESULTS_DIR.exists():
#         return jsonify({"folders": []})
#     folders_info = []
#     for folder in RESULTS_DIR.iterdir():
#         if folder.is_dir():
#             pid = folder.name
#             files = [f.name for f in folder.iterdir() if f.is_file()]
#             folders_info.append({
#                 "process_id": pid,
#                 "files": files
#             })
#     return jsonify({"folders": folders_info})

@app.route("/api/history", methods=["GET"])
def get_history():
    if not os.path.exists(RESULTS_DIR):
        return jsonify({"folders": []})

    folders_info = []
    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)

            # 取得資料夾最後修改時間
            timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(folder_path)))

            folders_info.append({
                "process_id": folder_name,
                "files": files,
                "timestamp": timestamp
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


@app.route("/result/<process_id>/<filename>", methods=["GET"])
def download_result(process_id, filename):
    file_path = RESULTS_DIR / process_id / filename
    if not file_path.exists():
        return abort(404, description="檔案不存在")
    return send_from_directory(file_path.parent, file_path.name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)