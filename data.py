import tensorflow as tf
import numpy as np
import pickle
import os

def unpickle(file):
    """Pickle dosyasını açar ve içeriğini döndürür."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_and_preprocess_data(data_dir="cifar-100-data"):
    """
    CIFAR-100 veri setini yerel dosyalardan yükler ve ön işler.
    Dosyaların 'data_dir' klasöründe olması beklenir.

    Args:
        data_dir (str): 'train', 'test', ve 'meta' dosyalarını içeren klasör.

    Returns:
        Tuple: (x_train, y_train), (x_test, y_test)
               Eğitim ve test verileri ile etiketleri.
    """
    # Eğitim verisini yükle
    train_file = os.path.join(data_dir, 'train')
    train_dict = unpickle(train_file)
    x_train = train_dict[b'data']
    y_train = np.array(train_dict[b'fine_labels'])

    # Test verisini yükle
    test_file = os.path.join(data_dir, 'test')
    test_dict = unpickle(test_file)
    x_test = test_dict[b'data']
    y_test = np.array(test_dict[b'fine_labels'])

    # Veriyi yeniden şekillendir
    # Gelen veri (N, 3072) formatında. Bunu (N, 3, 32, 32) ve sonra (N, 32, 32, 3) yapmalıyız.
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Görüntü piksellerini 0-1 aralığına normalize et
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Etiketleri one-hot encoding formatına dönüştür
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    # Fonksiyonun doğru çalışıp çalışmadığını test et
    # 'cifar-100-data' klasörünün var olduğundan ve dosyaları içerdiğinden emin olun
    if os.path.exists("cifar-100-data") and os.path.exists("cifar-100-data/train"):
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
        print("Eğitim verisi şekli:", x_train.shape)
        print("Eğitim etiketleri şekli:", y_train.shape)
        print("Test verisi şekli:", x_test.shape)
        print("Test etiketleri şekli:", y_test.shape)
        print("Veri başarıyla yüklendi ve işlendi.")
    else:
        print("Hata: 'cifar-100-data' klasörü bulunamadı veya 'train' dosyası eksik.")
        print("Lütfen 'train', 'test' ve 'meta' dosyalarınızı bu klasöre koyun.")
