import os
import joblib
from sklearn.naive_bayes import MultinomialNB

# Huấn luyện mô hình Naïve Bayes
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Lưu mô hình
def save_model(model, file_path):
    joblib.dump(model, file_path)

if __name__ == "__main__":
    # Lấy đường dẫn tuyệt đối đến thư mục hiện tại (src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Đường dẫn tuyệt đối đến file dataset đã xử lý
    processed_data_path = os.path.join(base_dir, "..", "data", "processed_dataset.pkl")
    
    # Đảm bảo file tồn tại trước khi load
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {processed_data_path}")
    
    # Load dữ liệu đã xử lý
    X_train, X_test, y_train, y_test = joblib.load(processed_data_path)
    
    # Huấn luyện mô hình
    model = train_model(X_train, y_train)
    
    # Đường dẫn tuyệt đối để lưu mô hình
    model_path = os.path.join(base_dir, "..", "models", "naive_bayes_model.pkl")
    
    # Lưu mô hình
    save_model(model, model_path)
    print(f"✅ Mô hình đã được lưu vào: {model_path}")
