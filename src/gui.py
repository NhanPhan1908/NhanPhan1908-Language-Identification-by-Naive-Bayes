import os
import sys
import joblib
import tkinter as tk
from tkinter import messagebox

# 🛠 Định nghĩa đường dẫn tuyệt đối
BASE_DIR = r"E:\NLP"  # Thay bằng đường dẫn thực tế của bạn
MODEL_PATH = os.path.join(BASE_DIR, "models", "naive_bayes_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# 🛠 Hàm tải mô hình và vectorizer
def load_model_and_vectorizer():
    print(f"🔍 Kiểm tra model tại: {MODEL_PATH}")  
    print(f"🔍 Kiểm tra vectorizer tại: {VECTORIZER_PATH}")  

    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("❌ Lỗi", f"Không tìm thấy mô hình tại:\n{MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(VECTORIZER_PATH):
        messagebox.showerror("❌ Lỗi", f"Không tìm thấy vectorizer tại:\n{VECTORIZER_PATH}")
        sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Mô hình và vectorizer đã tải thành công!")
        return model, vectorizer
    except Exception as e:
        messagebox.showerror("❌ Lỗi", f"Lỗi khi tải mô hình:\n{str(e)}")
        sys.exit(1)

# 🛠 Hàm dự đoán ngôn ngữ
def predict_language(event=None):
    text = text_entry.get().strip()
    if not text:
        messagebox.showwarning("⚠ Cảnh báo", "Vui lòng nhập văn bản để dự đoán!")
        return

    try:
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]
        result_label.config(text=f"🔎 Ngôn ngữ dự đoán: {prediction}", fg="green")
    except Exception as e:
        messagebox.showerror("❌ Lỗi", f"Lỗi khi dự đoán:\n{str(e)}")

# 🛠 Tạo giao diện GUI
root = tk.Tk()
root.title("🌍 Language Detection")
root.geometry("450x250")
root.resizable(False, False)  # Khóa kích thước cửa sổ

tk.Label(root, text="✍ Nhập văn bản:", font=("Arial", 12)).pack(pady=5)

text_entry = tk.Entry(root, width=55)
text_entry.pack(pady=5)
text_entry.bind("<Return>", predict_language)  # Nhấn Enter để dự đoán

predict_button = tk.Button(root, text="🔍 Dự đoán", command=predict_language, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white")
predict_button.pack(pady=10)

result_label = tk.Label(root, text="🔎 Ngôn ngữ dự đoán: ???", font=("Arial", 12, "bold"), fg="blue")
result_label.pack(pady=10)

# 🛠 Load mô hình và vectorizer
model, vectorizer = load_model_and_vectorizer()

root.mainloop()
