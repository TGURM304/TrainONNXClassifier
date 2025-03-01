import time
import tf2onnx
import onnx
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import Counter

os.makedirs("training_output", exist_ok=True)

train_dir = 'datasets/train'
val_dir = 'datasets/val'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=9,
    class_mode='categorical',
)

train_labels = train_generator.labels
val_labels = val_generator.labels

train_class_counts = dict(Counter(train_labels))
val_class_counts = dict(Counter(val_labels))

class_index_to_name = {v: k for k, v in train_generator.class_indices.items()}
train_class_counts = {class_index_to_name[key]: value for key, value in train_class_counts.items()}
val_class_counts = {class_index_to_name[key]: value for key, value in val_class_counts.items()}

class_labels = ['1', '2', '3', '4', 'base', 'qsz', 'sb', 'null']

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

start_time = time.time()

class CustomCheckpoint(Callback):
    def __init__(self, save_path="training_output", best_model_path="best.onnx"):
        super(CustomCheckpoint, self).__init__()
        self.save_path = save_path
        self.best_model_path = best_model_path
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0)
        onnx_filename = f"{self.save_path}/new.onnx"

        onnx_model, _ = tf2onnx.convert.from_keras(self.model)
        onnx.save_model(onnx_model, onnx_filename)
        print(f"Epoch {epoch+1}: Saved {onnx_filename}")

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            onnx.save_model(onnx_model, "./training_output/"+self.best_model_path)
            print(f"Epoch {epoch+1}: New best model saved as {self.best_model_path} with val_accuracy {val_acc:.4f}")

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

checkpoint_callback = CustomCheckpoint(save_path="training_output", best_model_path="best.onnx")

history = model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator,
    callbacks=[checkpoint_callback, lr_scheduler]
)

end_time = time.time()
training_duration = end_time - start_time

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(list(train_class_counts.keys()), list(train_class_counts.values()), color='blue')
axes[0, 0].set_title('Training Dataset Class Distribution')
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Count')

axes[0, 1].bar(list(val_class_counts.keys()), list(val_class_counts.values()), color='orange')
axes[0, 1].set_title('Validation Dataset Class Distribution')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Count')

axes[1, 0].plot(history.history['loss'], label='Train Loss')
axes[1, 0].plot(history.history['val_loss'], label='Val Loss')
axes[1, 0].set_title('Training and Validation Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

axes[1, 1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1, 1].set_title('Training and Validation Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('./training_output/training_report.png')

with open('./training_output/training_info.txt', 'w') as f:
    import io
    f.write(f"训练时长: {training_duration:.2f} 秒\n")
    f.write(f"训练集样本总数: {train_generator.samples}\n")
    f.write(f"验证集样本总数: {val_generator.samples}\n")
    f.write(f"训练批次大小: {train_generator.batch_size}\n")
    f.write(f"验证批次大小: {val_generator.batch_size}\n")
    f.write(f"训练轮数: {len(history.history['loss'])}\n")
    model_summary = io.StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    f.write(f"模型概况:\n{model_summary.getvalue()}\n")
    f.write("\n训练参数:\n")
    f.write(f"优化器: SGD\n")
    f.write(f"损失函数: Categorical Crossentropy\n")
    f.write(f"评估指标: Accuracy\n")
    f.write("\n模型输入输出信息:\n")
    f.write(f"输入: {model.input.shape}\n")
    f.write(f"输出: {model.output.shape}\n")
    f.write(f"输入数据类型: {model.input.dtype}\n")
    f.write(f"输出数据类型: {model.output.dtype}\n")

print("训练完成，最佳模型已保存为 best.onnx ✅")
