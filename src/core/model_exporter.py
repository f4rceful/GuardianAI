import torch
import os
from transformers import AutoTokenizer, AutoModel
import onnx
import onnxruntime as ort
import numpy as np

def export_to_onnx(model_name="cointegrated/rubert-tiny2", output_path="rubert_tiny2.onnx"):
    print(f"Загрузка модели: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Dummy input
    text = "Привет, это тестовое сообщение для экспорта."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Export
    print(f"Экспорт в {output_path}...")
    torch.onnx.export(
        model, 
        (inputs['input_ids'], inputs['attention_mask']), # Tuple of inputs
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
        },
        opset_version=14
    )
    print("Экспорт завершен.")
    
    # Verification
    print("Проверка ONNX модели...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Inference Test
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Compare with PyTorch
    with torch.no_grad():
        torch_out = model(**inputs)
    
    # Check max difference
    diff = np.max(np.abs(torch_out.last_hidden_state.numpy() - ort_outs[0]))
    print(f"Максимальная разница между PyTorch и ONNX: {diff}")
    
    if diff < 1e-4:
        print("[OK] ONNX модель валидна и точна.")
    else:
        print("[WARNING] Разница больше ожидаемой!")

if __name__ == "__main__":
    export_to_onnx()
