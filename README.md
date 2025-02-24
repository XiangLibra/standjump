# 立定跳遠

## 準備套件

``` bash
git clone https://github.com/tensorflow/examples.git
```

## 下載movenet模型

```bash
wget -q -O movenet_thunder.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite
```

## 下載python套件（用python3.10）
```bash
pip install requirements.txt
```

## 執行程式
```bash
flask run --host 0.0.0.0
``
