# ONNX и TensorRT: Оптимизация инференса ResNet-18

## Возможности
- Обучение ResNet-18
- Экспорт модели в ONNX
- Компиляция Torch-TensorRT (не получилось)
- Сравнение производительности PyTorch vs ONNX

## Запуск
```bash
# Установка
pip install torch torchvision onnx onnxruntime numpy torch_tensorrt

# Обучение
python trainer.py

# Экспорт ONNX
python core/torch_onnx.py

# Сравнение
python core/compare.py