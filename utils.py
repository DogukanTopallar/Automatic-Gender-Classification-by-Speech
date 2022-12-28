# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 00:00:18 2022

@author: DOGUKAN
"""
import pandas as pd #dataset işlemleri
import numpy as np
import os #dizin işlemleri & işletim sistemi işlemleri
import tqdm
from sklearn.model_selection import train_test_split #test eğitim ayrımı
from tensorflow.keras.layers import Dense, Dropout #LSTM
from tensorflow.keras.models import Sequential


#etiketleri 1-0 olarak değiştirecek dictionary oluşturuluyor.
label2int = {
    "male": 1,
    "female": 0
}

def load_data(vector_length=128):
    if not os.path.isdir(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results"):
        os.mkdir(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results")
        #os.path.isdir -> parametre olarak gönderilen path'in dizin olup olmadığını kontrol ediyor.
        #os.mkdir -> yeni dizin oluşturmayı sağlıyor.
        
    if os.path.isfile(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results\features.npy") and os.path.isfile(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results\labels.npy"):
        #os.path.isfile -> verilen yolun gerçek bir dosya yolu olup olmadığını kontrol eder.
        
        X = np.load(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results\features.npy")
        y = np.load(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\results\labels.npy")
        return X, y
    
    df = pd.read_csv(r"C:\Users\Dogukan\Desktop\DERSLER\BM YÜKSEK LİSANS\ÇALIŞMALAR\ANACONDA-SPYDER\GenderClassificationByVoice\Automatic-Gender-Classification-by-Speech\balanced-all.csv")

    
    n_samples = len(df)
    # Toplam örnek sayısını getirir
    
    n_male_samples = len(df[df['gender'] == 'male'])
    # Toplam erkek örnek sayısını getirir
    
    n_female_samples = len(df[df['gender'] == 'female'])
    # Toplam kadın örnek sayısını getirir
    
    print("Toplam Örnek Sayısı:", n_samples)
    print("Toplam Erkek Örnek Sayısı:", n_male_samples)
    print("Toplam Kadın Örnek Sayısı:", n_female_samples)
    
    
    X = np.zeros((n_samples, vector_length))
    #Belirlenen vektör uzunluğunda tüm ses örnekleri için boş bir dizi oluşturulur.
    
    y = np.zeros((n_samples, 1))
    #Tüm ses etiketleri için boş bir dizi başlatılır.
    
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        features = np.load(filename)
        X[i] = features
        y[i] = label2int[gender]
        #ekran görüntüsü var
        
        np.save("results/features", X)
        np.save("results/labels", y)
        return X, y

def split_data(X, y, test_size=0.1, valid_size=0.1):
#%80 train_set, %10 test, %10 valid(model performansını doğrulamak için kullanılır) olarak ayrılır.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    #Ham veri içerisinden %10'u test olacak şekilde train ve test verilerini ayırır.
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    #Ayrılan veriler içerisinden %10'u valid(model doğrulamak için) olacak şekilde train ve valid verilerini ayırır.
    
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }
    # Bir dictionary olarak ayrılmış verileri saklar.

X, y = load_data()
data = split_data(X, y, test_size=0.1, valid_size=0.1)
   
#relu -> sıfırın altındaki değerlere sıfır, üstündeki değerlere kendi değerini atar.
#sigmoid -> elde edilen değerleri 0-1 arasında bir değere tutturur.
#Batch Size -> Küme büyüklüğü (batch_size) bir seferde yapay sinir ağını eğitmek için kullanılacak örnek sayısını belirtir.
#Epoch -> Devir sayısı.
#Dense -> Katmandaki düğüm sayısıdır. Modelin esnekliğini artırır.
def create_model(vector_length=128):
    """256 birimden 64'e 5 gizli yoğun katman. """
    model = Sequential()
    #sıralı model oluşturulur.
    
    model.add(Dense(256, input_shape=(vector_length,), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    
    # sigmoid aktivasyon fonksiyonuna sahip bir çıkış nöronu, 0 kadın, 1 erkek anlamına gelir.
    model.add(Dense(1, activation="sigmoid"))
    # ikili sınıflandırma oldugundan cross entropy kullanıldı. Doğruluk bakımından incelendi. 
    # Adam, eğitim verilerine dayalı yinelemeli ağ ağırlıklarını güncellemek için klasik stokastik gradyan iniş prosedürü yerine kullanılabilen bir optimizasyon algoritmasıdır.
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # modelin özetini yazdırır.
    model.summary()
    return model







