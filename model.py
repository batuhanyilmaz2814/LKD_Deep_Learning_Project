from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape, num_classes):
    
    #Sıfırdan bir CNN modeli oluşturur.

    """
    input_shape (tuple): Girdi görüntüsünün şekli (224, genişlik, kanal).
    num_classes (int): Sınıf sayısı.
    
    Returns:
        tensorflow.keras.Model: Derlenmemiş Keras modeli.
    """
    model = Sequential([
        # 1. Konvolüsyon Katmanı
        Conv2D(96, (11, 11), activation='relu', input_shape=input_shape),
        MaxPooling2D((3, 3), strides=2),

        # 2. Konvolüsyon Katmanı
        Conv2D(256, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((3, 3), strides=2),

        # 3. Konvolüsyon Katmanı
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((3, 3), strides=2),

        # Düzleştirme Katmanı
        Flatten(),

        # Tam Bağlantılı Katman
        Dense(4096, activation='relu'),
        Dropout(0.5),  # Overfitting'i önlemek için

        Dense(4096, activation='relu'),
        Dropout(0.5),  # Overfitting'i önlemek için

        # Çıkış Katmanı
        Dense(num_classes, activation='softmax') # Sınıflandırma için softmax
    ])

    return model

if __name__ == '__main__':
    # Modelin doğru oluşturulup oluşturulmadığını test et
    img_shape = (32, 32, 3)
    n_classes = 100
    model = create_model(img_shape, n_classes)
    model.summary() # Modelin özetini yazdır
