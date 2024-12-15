import time
import tf2onnx
import onnx
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 设置数据路径
train_dir = 'datasets/train'
val_dir = 'datasets/val'

# 图像预处理和数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 图像归一化
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# 读取训练和验证数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # 图像尺寸调整为 64x64
    batch_size=16,
    class_mode='categorical',  # 多分类
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),  # 图像尺寸调整为 64x64
    batch_size=16,
    class_mode='categorical',  # 多分类
)

# 获取每个类别的样本数量
train_labels = train_generator.labels
val_labels = val_generator.labels

train_class_counts = dict(Counter(train_labels))
val_class_counts = dict(Counter(val_labels))

# 获取类别索引到类别名称的映射
class_index_to_name = {v: k for k, v in train_generator.class_indices.items()}

# 将类索引转换为类名称
train_class_counts = {class_index_to_name[key]: value for key, value in train_class_counts.items()}
val_class_counts = {class_index_to_name[key]: value for key, value in val_class_counts.items()}

# 将 dict_keys 和 dict_values 转换为 list 类型
train_class_names = list(train_class_counts.keys())
train_class_values = list(train_class_counts.values())

val_class_names = list(val_class_counts.keys())
val_class_values = list(val_class_counts.values())

# 分类标签顺序
class_labels = ['1', '2', '3', '4', 'base', 'qsz', 'sb', 'null']

# 构建卷积神经网络（CNN）
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
    layers.Dropout(0.5),  # 防止过拟合
    layers.Dense(8, activation='softmax')  # 类别
])

# 编译模型
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# 模型概况
model.summary()

# 创建学习率调度器
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',   # 监控验证集的损失
    factor=0.5,           # 每次减少学习率的倍数
    patience=5,           # 如果验证集损失没有改善，等待多少轮再调整学习率
    verbose=1,            # 打印学习率调整的信息
    min_lr=1e-6           # 设置学习率的下限
)

# 记录训练开始时间
start_time = time.time()

# 训练模型
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

# 记录训练结束时间
end_time = time.time()
training_duration = end_time - start_time

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 训练集柱形图
axes[0, 0].bar(train_class_names, train_class_values, color='blue')
axes[0, 0].set_title('Training Dataset Class Distribution')
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticks(range(len(train_class_names)))
axes[0, 0].set_xticklabels(class_labels)

# 验证集柱形图
axes[0, 1].bar(val_class_names, val_class_values, color='orange')
axes[0, 1].set_title('Validation Dataset Class Distribution')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xticks(range(len(val_class_names)))
axes[0, 1].set_xticklabels(class_labels)

# 绘制训练和验证的损失曲线
axes[1, 0].plot(history.history['loss'], label='Train Loss')
axes[1, 0].plot(history.history['val_loss'], label='Val Loss')
axes[1, 0].set_title('Training and Validation Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

# 绘制训练和验证的准确度曲线
axes[1, 1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1, 1].set_title('Training and Validation Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()

# 保存整个图像到本地文件
plt.tight_layout()
plt.savefig('training_report.png')

# 将 Keras 模型转换为 ONNX 格式
onnx_model, _ = tf2onnx.convert.from_keras(model)

# 保存 ONNX 模型
onnx.save_model(onnx_model, "best.onnx")

# 保存训练信息到TXT文件
with open('training_info.txt', 'w') as f:
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
    f.write(f"优化器: Adam\n")
    f.write(f"损失函数: Categorical Crossentropy\n")
    f.write(f"评估指标: Accuracy\n")
    f.write("\n模型输入输出信息:\n")
    f.write(f"输入: {model.input.shape}\n")
    f.write(f"输出: {model.output.shape}\n")
    f.write(f"输入数据类型: {model.input.dtype}\n")
    f.write(f"输出数据类型: {model.output.dtype}\n")
