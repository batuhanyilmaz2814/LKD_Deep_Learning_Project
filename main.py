import tensorflow as tf
from data import load_and_preprocess_data
from model import create_model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Hiperparametreler
NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 96
EPOCHS = 10 # Epoch sayısını artırdık, EarlyStopping en iyi noktada durduracak

def main():
    """
    Ana proje akışı.
    """
    # 1. Veriyi yükle ve ön işle
    print("Veri yükleniyor ve ön işleniyor...")
    (x_train, y_train), (x_test, y_test), datagen = load_and_preprocess_data()
    print("Veri başarıyla yüklendi.")

    # 2. Modeli oluştur
    print("Model oluşturuluyor...")
    model = create_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    print("Model başarıyla oluşturuldu.")

    # 3. Modeli derle
    # Optimizatör, kayıp fonksiyonu ve metrikleri belirliyoruz.
    # RMSprop, 0.001 öğrenme oranıyla daha iyi sonuç verdi.
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # Çok sınıflı sınıflandırma için
        metrics=['accuracy']
    )
    model.summary()

    # 4. Modeli eğit
    print("\nModel eğitimi başlıyor...")
    # Veri artırma ile eğitim için model.fit_generator veya model.fit kullanın
    # validation_split yerine validation_data sağlamak daha doğrudur.
    # Veri setinden bir doğrulama seti ayıralım
    val_split = int(x_train.shape[0] * 0.9)
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]

    # Callback'leri tanımla
    # 1. EarlyStopping: Modelin performansı artmadığında eğitimi durdurur.
    early_stopping = EarlyStopping(
        monitor='val_loss', # İzlenecek metrik: doğrulama kaybı
        patience=10,        # Performansın 10 epoch boyunca iyileşmesini bekle
        verbose=1,          # Durdurulduğunda mesaj göster
        restore_best_weights=True # En iyi ağırlıkları geri yükle
    )

    # 2. ReduceLROnPlateau: Performans platoya ulaştığında öğrenme oranını düşürür.
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', # İzlenecek metrik
        factor=0.2,         # Öğrenme oranını düşürme faktörü (new_lr = lr * factor)
        patience=5,         # 5 epoch boyunca iyileşme olmazsa öğrenme oranını düşür
        min_lr=0.00001,     # Öğrenme oranının düşebileceği minimum değer
        verbose=1
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr] # Callback'leri eğitime ekle
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
