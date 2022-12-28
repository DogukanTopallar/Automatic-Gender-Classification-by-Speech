# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:07:15 2022

@author: DOGUKAN
"""

#import os
from tensorflow.keras.callbacks import  TensorBoard, EarlyStopping #ModelCheckpoint
#Ölçümleri görüntülemek için kullanılır.

from utils import load_data, split_data, create_model
#utils'de oluşturulan fonksiyonlar çağırılır.

X, y = load_data()
#features ve labels çağırılır.

data = split_data(X, y, test_size=0.1, valid_size=0.1)
# train, validation, test verileri ayrılır.

model = create_model()
#model inşa edilir.

tensorboard = TensorBoard(log_dir="logs")
# tenserboard kullanarak logları kaydeder.

early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)
#Son 5 val_loss değeri arttıkça eğitimi durdurur.

batch_size = 64
# parametre güncellemesinin gerçekleştiği ağa verilen alt örneklerin sayısıdır. Bir seferde kullanılacak örnek sayısı.

epochs = 100
#eğitim adımları & devir sayısı


model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])
#Modeli eğitim setini kullanarak eğitir, valid setini kullanarak doğrulama yapar.

# save the model to a file
model.save("results/model.h5")

# evaluating the model using the testing set
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")