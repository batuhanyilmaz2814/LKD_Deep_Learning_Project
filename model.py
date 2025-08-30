from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

def create_model(input_shape, num_classes):
    """
    CIFAR-100 için optimize edilmiş bir CNN modeli oluşturur.
    Daha küçük filtreler, Batch Normalization ve artan Dropout kullanır.
    """
    model = Sequential([
        # Blok 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Blok 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Blok 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Düzleştirme ve Yoğun Katmanlar
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    # Modelin doğru oluşturulup oluşturulmadığını test et
    img_shape = (32, 32, 3)
    n_classes = 100
    model = create_model(img_shape, n_classes)
    model.summary() # Modelin özetini yazdır
