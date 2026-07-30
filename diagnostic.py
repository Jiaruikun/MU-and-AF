# 创建一个诊断脚本 diagnostic.py
import joblib
import os

MODEL_PATH = r"G:\APP\MU\tabpfn_model.joblib"

print("=== Model File Diagnostic ===")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"Model data type: {type(model_data)}")
        print(f"Model data keys: {model_data.keys() if hasattr(model_data, 'keys') else 'No keys attribute'}")
        print(f"Model data contents: {model_data}")
        
        # 如果是字典，打印所有键值对
        if isinstance(model_data, dict):
            for key, value in model_data.items():
                print(f"Key: {key}, Type: {type(value)}, Value: {value}")
        
    except Exception as e:
        print(f"Error loading model: {e}")