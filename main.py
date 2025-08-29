import tensorflow as tf
from data import load_and_preprocess_data
from model import create_model

# Hiperparametreler
NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 64
EPOCHS = 25 # Başlangıç için daha az epoch, daha sonra artırılabilir

def main():
    """
    Ana proje akışı.
    """
    # 1. Veriyi yükle ve ön işle
    print("Veri yükleniyor ve ön işleniyor...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("Veri başarıyla yüklendi.")

    # 2. Modeli oluştur
    print("Model oluşturuluyor...")
    model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    print("Model başarıyla oluşturuldu.")

    # 3. Modeli derle
    # Optimizatör, kayıp fonksiyonu ve metrikleri belirliyoruz.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Çok sınıflı sınıflandırma için
        metrics=['accuracy']
    )
    model.summary()

    # 4. Modeli eğit
    print("\nModel eğitimi başlıyor...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1 # Eğitim verisinin %10'unu doğrulama için ayır
    )
    print("Model eğitimi tamamlandı.")

    # 5. Modeli test seti ile değerlendir
    print("\nModel test seti üzerinde değerlendiriliyor...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Seti Doğruluğu: {test_accuracy * 100:.2f}%")
    print(f"Test Seti Kaybı: {test_loss:.4f}")

    # Modeli kaydedebilirsiniz
    # model.save('cifar100_model.h5')
    # print("\nModel 'cifar100_model.h5' olarak kaydedildi.")


if __name__ == '__main__':
    main()
