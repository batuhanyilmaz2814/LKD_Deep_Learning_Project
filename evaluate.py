import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_class_names(data_dir="cifar-100-data"):
    """CIFAR-100 sınıf isimlerini yükler."""
    try:
        meta_file = os.path.join(data_dir, 'meta')
        with open(meta_file, 'rb') as f:
            meta_dict = pickle.load(f, encoding='bytes')
        
        # Fine label names (100 sınıf)
        fine_label_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
        return fine_label_names
    except:
        # Eğer meta dosyası bulunamazsa genel isimler kullan
        return [f'Class_{i}' for i in range(100)]

def predict_and_show_results(model, x_test, y_test, num_examples=10, data_dir="cifar-100-data"):
    """
    Modelin tahminlerini gösterir ve sonuçları analiz eder.
    
    Args:
        model: Eğitilmiş model
        x_test: Test görüntüleri
        y_test: Test etiketleri (one-hot encoded)
        num_examples: Gösterilecek örnek sayısı
        data_dir: Veri klasörü
    """
    # Sınıf isimlerini yükle
    class_names = load_class_names(data_dir)
    
    # Tahminleri yap
    print("Tahminler yapılıyor...")
    predictions = model.predict(x_test, verbose=0)
    
    # One-hot encoded'dan normal etiketlere çevir
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Doğruluk hesapla
    accuracy = np.mean(true_labels == predicted_labels)
    print(f"Test Doğruluğu: {accuracy*100:.2f}%")
    
    # Rastgele örnekler seç
    random_indices = np.random.choice(len(x_test), num_examples, replace=False)
    
    # Sonuçları görselleştir
    plt.figure(figsize=(15, 3*((num_examples+4)//5)))
    
    for i, idx in enumerate(random_indices):
        plt.subplot((num_examples+4)//5, 5, i+1)
        
        # Görüntüyü göster
        plt.imshow(x_test[idx])
        plt.axis('off')
        
        # Tahmin ve gerçek etiket
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[predicted_labels[idx]]
        confidence = predictions[idx][predicted_labels[idx]] * 100
        
        # Doğru/yanlış rengini belirle
        color = 'green' if true_labels[idx] == predicted_labels[idx] else 'red'
        
        plt.title(f'Gerçek: {true_class}\n'
                 f'Tahmin: {pred_class}\n'
                 f'Güven: {confidence:.1f}%', 
                 color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('predictions_sample.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Tahmin örnekleri 'predictions_sample.png' dosyasına kaydedildi.")

def analyze_predictions(model, x_test, y_test, data_dir="cifar-100-data"):
    """
    Tahminleri detaylı analiz eder ve en çok karıştırılan sınıfları gösterir.
    """
    class_names = load_class_names(data_dir)
    
    # Tahminleri yap
    predictions = model.predict(x_test, verbose=0)
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print("\n" + "="*60)
    print("DETAYLI TAHMİN ANALİZİ")
    print("="*60)
    
    # En yüksek güvenle doğru tahmin edilen örnekler
    correct_mask = (true_labels == predicted_labels)
    correct_confidences = np.max(predictions[correct_mask], axis=1)
    
    if len(correct_confidences) > 0:
        top_correct_idx = np.where(correct_mask)[0][np.argmax(correct_confidences)]
        print(f"\nEN GÜVENLE DOĞRU TAHMİN:")
        print(f"Sınıf: {class_names[true_labels[top_correct_idx]]}")
        print(f"Güven: {np.max(predictions[top_correct_idx])*100:.2f}%")
    
    # En yüksek güvenle yanlış tahmin edilen örnekler
    incorrect_mask = (true_labels != predicted_labels)
    if np.any(incorrect_mask):
        incorrect_confidences = np.max(predictions[incorrect_mask], axis=1)
        top_incorrect_idx = np.where(incorrect_mask)[0][np.argmax(incorrect_confidences)]
        print(f"\nEN GÜVENLE YANLIŞ TAHMİN:")
        print(f"Gerçek: {class_names[true_labels[top_incorrect_idx]]}")
        print(f"Tahmin: {class_names[predicted_labels[top_incorrect_idx]]}")
        print(f"Güven: {np.max(predictions[top_incorrect_idx])*100:.2f}%")
    
    # Sınıf bazında doğruluk
    class_accuracies = []
    for class_idx in range(100):
        class_mask = (true_labels == class_idx)
        if np.any(class_mask):
            class_acc = np.mean(predicted_labels[class_mask] == class_idx)
            class_accuracies.append((class_idx, class_acc))
    
    # En iyi ve en kötü performans gösteren sınıflar
    class_accuracies.sort(key=lambda x: x[1])
    
    print(f"\nEN KÖTÜ 5 SINIF:")
    for i in range(min(5, len(class_accuracies))):
        class_idx, acc = class_accuracies[i]
        print(f"{class_names[class_idx]}: {acc*100:.1f}%")
    
    print(f"\nEN İYİ 5 SINIF:")
    for i in range(max(0, len(class_accuracies)-5), len(class_accuracies)):
        class_idx, acc = class_accuracies[i]
        print(f"{class_names[class_idx]}: {acc*100:.1f}%")
    
    print("="*60)

if __name__ == '__main__':
    # Bu dosya tek başına çalıştırılırsa test verileri ile örnek gösterir
    print("evaluate.py modülü yüklendi. main.py üzerinden kullanın.")