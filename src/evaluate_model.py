import os
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Định nghĩa đường dẫn tuyệt đối
BASE_DIR = r"E:\NLP"  # Cập nhật đường dẫn của bạn tại đây
MODEL_PATH = os.path.join(BASE_DIR, "models", "naive_bayes_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed_dataset.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")

# Kiểm tra và tạo thư mục results nếu chưa có
os.makedirs(RESULTS_DIR, exist_ok=True)

# Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    # Kiểm tra sự tồn tại của file mô hình và dataset
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy mô hình tại {MODEL_PATH}")
        exit()

    if not os.path.exists(DATASET_PATH):
        print(f"LỖI: Không tìm thấy dataset tại {DATASET_PATH}")
        exit()

    # Load mô hình và dữ liệu
    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = joblib.load(DATASET_PATH)

    # Đánh giá mô hình
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Lưu kết quả vào file
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")

    print(f"Kết quả đã được lưu vào {RESULTS_PATH}")
