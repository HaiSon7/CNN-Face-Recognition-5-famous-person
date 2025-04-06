import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Lớp quản lý các tham số
class Config:
    def __init__(self):
        # Đường dẫn dữ liệu
        self.TRAIN_DATA = 'image/data/train'
        self.TEST_DATA = 'image/data/test'
        
        # Tham số mô hình
        self.IMG_SIZE = (70, 70)
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.VALIDATION_SPLIT = 0.2
        
        # Tham số augmentation
        self.ROTATION_RANGE = 20
        self.WIDTH_SHIFT_RANGE = 0.2
        self.HEIGHT_SHIFT_RANGE = 0.2
        self.SHEAR_RANGE = 0.2
        self.ZOOM_RANGE = 0.2
        self.HORIZONTAL_FLIP = True
        
        # Labels và one-hot encoding
        self.LABELS = ['Bill Gates', 'Elon Musk', 'Jeff Bezos', 'Mark Zuckerberg', 'Steve Jobs']
        self.LABEL_DICT = {
            "bill_gates": [1, 0, 0, 0, 0],
            "elon_musk": [0, 1, 0, 0, 0],
            "jeff_bezos": [0, 0, 1, 0, 0],
            "mark_zuckerberg": [0, 0, 0, 1, 0],
            "steve_jobs": [0, 0, 0, 0, 1]
        }
        
        # Tên thư mục tương ứng với các nhãn
        self.FOLDER_TO_INDEX = {
            "bill_gates": 0,
            "elon_musk": 1,
            "jeff_bezos": 2,
            "mark_zuckerberg": 3,
            "steve_jobs": 4
        }
        
        # Đường dẫn lưu mô hình
        self.MODEL_PATH = "models/face_recognition_model.keras"
        self.BEST_MODEL_PATH = "models/best_face_recognition_model.keras"

        self.TRANSFER_Model_path = "models/transfer_model.keras"
        self.TRANSFER_Best_Model_path = "models/best_transfer_model.keras"

# Lớp xử lý dữ liệu
class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_and_preprocess_data(self, dir_data):
        """
        Tải và tiền xử lý dữ liệu từ thư mục
        
        Args:
            dir_data (str): Đường dẫn thư mục dữ liệu
            
        Returns:
            tuple: (x_data, y_data) - Mảng ảnh và nhãn tương ứng
        """
        x_data = []
        y_data = []
        
        for item in os.listdir(dir_data):
            item_path = os.path.join(dir_data, item)
            
            if not os.path.isdir(item_path):
                continue
                
            for filename in os.listdir(item_path):
                filename_path = os.path.join(item_path, filename)
                
                try:
                    # Đọc ảnh màu (BGR)
                    image = cv2.imread(filename_path)
                    
                    if image is None:
                        print(f"Không thể đọc ảnh: {filename_path}")
                        continue
                    
                    # Chuyển đổi từ BGR sang RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                    # Resize ảnh về kích thước mong muốn
                    resized_image = cv2.resize(image, self.config.IMG_SIZE)
                    
                    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
                    normalized_image = resized_image / 255.0
                    
                    # Thêm ảnh và nhãn tương ứng vào danh sách
                    x_data.append(normalized_image)
                    y_data.append(self.config.LABEL_DICT[item])
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {filename_path}: {str(e)}")
        
        return np.array(x_data), np.array(y_data)
    
    def split_data(self, x_data, y_data):
        """
        Chia dữ liệu thành tập train và validation
        
        Args:
            x_data (np.array): Mảng dữ liệu ảnh
            y_data (np.array): Mảng nhãn
            
        Returns:
            tuple: (x_train, x_val, y_train, y_val)
        """
        return train_test_split(
            x_data, y_data, 
            test_size=self.config.VALIDATION_SPLIT, 
            random_state=42,
            stratify=y_data  # Đảm bảo phân bố nhãn đồng đều
        )
    
    def create_data_generators(self, x_train, y_train, x_val, y_val):
        """
        Tạo data generators với augmentation cho tập train
        
        Args:
            x_train, y_train, x_val, y_val: Dữ liệu train và validation
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Tạo generator cho augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP
        )
        
        # Generator cho validation (không cần augment)
        val_datagen = ImageDataGenerator()
        
        # Tạo generators
        train_generator = train_datagen.flow(
            x_train, y_train,
            batch_size=self.config.BATCH_SIZE
        )
        
        val_generator = val_datagen.flow(
            x_val, y_val,
            batch_size=self.config.BATCH_SIZE
        )
        
        return train_generator, val_generator

# Lớp xây dựng mô hình
class FaceRecognitionModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        
    def build_model(self):
        """
        Xây dựng kiến trúc mô hình CNN
        
        Returns:
            tensorflow.keras.Model: Mô hình đã định nghĩa
        """
 
        # Load MobileNetV2 without the classification layers
        # MobileNetV2 works well with smaller image sizes
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model to prevent its weights from being updated during training
        base_model.trainable = False
        
        # Create a new model on top of the base model
        model = models.Sequential([
            
            # Base model
            base_model,
            
            # Classification layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.config.LABEL_DICT), activation='softmax')
        ])
            
        
        # Biên dịch mô hình
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def get_callbacks(self):
        """
        Định nghĩa các callbacks cho quá trình huấn luyện
        
        Returns:
            list: Danh sách các callbacks
        """
        # Dừng sớm nếu mô hình không cải thiện
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        # Giảm learning rate khi mô hình không cải thiện
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Lưu mô hình tốt nhất
        model_checkpoint = ModelCheckpoint(
            self.config.TRANSFER_Best_Model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        return [early_stopping, reduce_lr, model_checkpoint]
    
    def train(self, train_generator, val_generator, steps_per_epoch, validation_steps):
        """
        Huấn luyện mô hình
        
        Args:
            train_generator: Generator cho tập train
            val_generator: Generator cho tập validation
            steps_per_epoch: Số bước mỗi epoch
            validation_steps: Số bước validation
            
        Returns:
            history: Lịch sử huấn luyện
        """
        # Tạo thư mục lưu mô hình nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.config.TRANSFER_Model_path), exist_ok=True)
        
        # Huấn luyện mô hình
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=self.get_callbacks()
        )
        
        # Lưu mô hình cuối cùng
        self.model.save(self.config.TRANSFER_Model_path)
        
        return history
    
    def evaluate(self, x_test, y_test):
        """
        Đánh giá mô hình trên tập test
        
        Args:
            x_test: Dữ liệu test
            y_test: Nhãn test
            
        Returns:
            dict: Kết quả đánh giá (loss, accuracy)
        """
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x_test):
        """
        Dự đoán nhãn cho dữ liệu test
        
        Args:
            x_test: Dữ liệu test
            
        Returns:
            np.array: Mảng các dự đoán
        """
        return self.model.predict(x_test)

# Lớp trực quan hóa kết quả
class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_training_history(self, history):
        """
        Vẽ biểu đồ quá trình huấn luyện
        
        Args:
            history: Lịch sử huấn luyện từ model.fit()
        """
        # Tạo figure với 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_transfer_model.png')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Vẽ ma trận nhầm lẫn
        
        Args:
            y_true: Nhãn thực tế
            y_pred: Nhãn dự đoán
        """
        # Chuyển đổi từ one-hot vectors sang class indices
        y_true_indices = np.argmax(y_true, axis=1)
        y_pred_indices = np.argmax(y_pred, axis=1)
        
        # Tính ma trận nhầm lẫn
        cm = confusion_matrix(y_true_indices, y_pred_indices)
        
        # Vẽ ma trận nhầm lẫn
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.config.LABELS,
            yticklabels=self.config.LABELS
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix_transfer_model.png')
        plt.show()
        
    def print_classification_report(self, y_true, y_pred):
        """
        In báo cáo phân loại
        
        Args:
            y_true: Nhãn thực tế
            y_pred: Nhãn dự đoán
        """
        # Chuyển đổi từ one-hot vectors sang class indices
        y_true_indices = np.argmax(y_true, axis=1)
        y_pred_indices = np.argmax(y_pred, axis=1)
        
        # In báo cáo phân loại
        print("\nClassification Report:")
        print(classification_report(
            y_true_indices, 
            y_pred_indices, 
            target_names=self.config.LABELS
        ))

# Hàm chính
def main():
    # Khởi tạo cấu hình
    config = Config()
    
    # Khởi tạo bộ xử lý dữ liệu
    data_processor = DataProcessor(config)
    
    # Tải dữ liệu huấn luyện và kiểm tra
    print("Đang tải dữ liệu huấn luyện...")
    x_data, y_data = data_processor.load_and_preprocess_data(config.TRAIN_DATA)
    print(f"Đã tải {len(x_data)} mẫu huấn luyện.")
    
    # Chia dữ liệu thành tập train và validation
    x_train, x_val, y_train, y_val = data_processor.split_data(x_data, y_data)
    print(f"Tập train: {len(x_train)} mẫu, Tập validation: {len(x_val)} mẫu")
    
    # Tải dữ liệu test
    print("Đang tải dữ liệu test...")
    x_test, y_test = data_processor.load_and_preprocess_data(config.TEST_DATA)
    print(f"Đã tải {len(x_test)} mẫu test.")
    
    # Tạo data generators
    train_generator, val_generator = data_processor.create_data_generators(
        x_train, y_train, x_val, y_val
    )
    
    # Khởi tạo mô hình
    print("Đang xây dựng mô hình...")
    model = FaceRecognitionModel(config)
    model.model.summary()
    
    # Tính số bước mỗi epoch
    steps_per_epoch = len(x_train) // config.BATCH_SIZE
    validation_steps = len(x_val) // config.BATCH_SIZE
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    history = model.train(train_generator, val_generator, steps_per_epoch, validation_steps)
    
    # Đánh giá mô hình trên tập test
    print("Đánh giá mô hình trên tập test...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Dự đoán trên tập test
    y_pred = model.predict(x_test)
    
    # Trực quan hóa kết quả
    visualizer = Visualizer(config)
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_test, y_pred)
    visualizer.print_classification_report(y_test, y_pred)
    
if __name__ == "__main__":
    main()