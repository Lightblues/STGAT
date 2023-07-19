import tensorflow as tf
from tensorflow import keras

# 准备训练数据和标签
train_images = [0,1,2,3,4]
train_labels = [0,1,2,3,4]
train_images = tf.reshape(train_images, (5, 1))
train_labels = tf.reshape(train_labels, (5, 1))
sample_weight = [1,2,3,4,5]
sample_weight = tf.reshape(sample_weight, (5, 1))

# 构建模型
model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(28, 28)),  # 将输入展平
    keras.layers.Dense(2, activation='relu'),  # 添加全连接层
    keras.layers.Dense(5, activation='softmax')  # 添加输出层
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 进行训练
model.fit(train_images, train_labels, sample_weight=sample_weight, epochs=10)

# 使用模型进行预测
test_images = [[1],[2]]
predictions = model.predict(test_images)
print(predictions)