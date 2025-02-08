import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Lấy đường dẫn tuyệt đối đến thư mục hiện tại (src/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn tuyệt đối đến thư mục data & models
data_dir = os.path.join(base_dir, "..", "data")
models_dir = os.path.join(base_dir, "..", "models")

# Đảm bảo thư mục tồn tại
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Đường dẫn dữ liệu đầu vào
data_path = os.path.join(data_dir, "language_detection.csv")

# Đường dẫn file output
processed_data_path = os.path.join(data_dir, "processed_dataset.pkl")
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")

# Hàm load dữ liệu
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.dropna()  # Xóa dòng trống nếu có
    return df['Text'], df['Language']

# Hàm tiền xử lý dữ liệu
def preprocess_data(texts, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, labels, vectorizer

if __name__ == "__main__":
    try:
        # Load dữ liệu từ CSV
        texts, labels = load_data(data_path)
        
        # Tiền xử lý dữ liệu
        X, y, vectorizer = preprocess_data(texts, labels)
        
        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lưu dữ liệu đã xử lý
        joblib.dump((X_train, X_test, y_train, y_test), processed_data_path)
        joblib.dump(vectorizer, vectorizer_path)

        print(f"✅ Dữ liệu đã được lưu vào: {processed_data_path}")
        print(f"✅ Vectorizer đã được lưu vào: {vectorizer_path}")

    except Exception as e:
        print(f"⚠️ Lỗi: {e}")
