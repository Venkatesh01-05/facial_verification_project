import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
real_folder=r"C:\Users\Venka\facial_verification_project\data\real_vs_fake\real-vs-fake\train\real"
fake_folder=r"C:\Users\Venka\facial_verification_project\data\real_vs_fake\real-vs-fake\train\fake"
files = os.listdir(real_folder)
print(len(files))
print(files[:5])
#file_path =os.path.join(real_folder,files[0])
#Image.open(file_path)

# %%
#get details of images
real_file=os.listdir(real_folder)[:5]
fake_file=os.listdir(fake_folder)[:5]
print(real_file)
print(fake_file)

# %%
fig,axes=plt.subplots(2,5,figsize=(15,6))
print(axes.shape)
for i,fname in enumerate(real_file):
    image_path=os.path.join(real_folder, fname)
    img=Image.open(image_path)
    axes[0,i].imshow(img)
for i,fname in enumerate(fake_file):
    image_path=os.path.join(fake_folder, fname)
    img=Image.open(image_path)
    axes[1,i].imshow(img)
    

# %%
import tensorflow as tf
train_folder=r"C:\Users\Venka\facial_verification_project\data\real_vs_fake\real-vs-fake\train"
train_ds=tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=(224,224),
    batch_size=32
)
valid_folder=r"C:\Users\Venka\facial_verification_project\data\real_vs_fake\real-vs-fake\valid"
val_ds=tf.keras.utils.image_dataset_from_directory(
    valid_folder,
    image_size=(224,224),
    batch_size=32
)

# %%
#for research puposes
"""research_model=tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)
research_model.trainable=False
rinputs=tf.keras.Input(shape=(224,224,3))
y=research_model(rinputs, training=False)
y=tf.keras.layers.GlobalAveragePooling2D()(y)
y=tf.keras.layers.Dropout(0.2)(y)
y=tf.keras.layers.Dense(1,activation="sigmoid")(y)
rmodel=tf.keras.Model(rinputs,y)
rmodel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
rmodel.summary()"""
base_model=tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)
base_model.trainable=False
inputs=tf.keras.Input(shape=(224,224,3))
x=base_model(inputs,training=False)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(1,activation="sigmoid")(x)

model=tf.keras.Model(inputs,x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# %%
import time
small_train=train_ds.take(20)
start=time.time()
#rmodel.fit(small_train,epochs=1)
model.fit(small_train,epochs=1)
end=time.time()
print("time for 20 batches:",end-start)

# %%
check_point=tf.keras.callbacks.ModelCheckpoint(
    filepath="best_efficientnet_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stop=tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# %%


history=model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=50,
                  callbacks=[check_point,early_stop]
                  )

# %%
test_folder=r"C:\Users\Venka\facial_verification_project\data\real_vs_fake\real-vs-fake\test"
test_ds=tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=(224,224),
    batch_size=32,
    shuffle=False
)
results=model.evaluate(test_dsepochs=10,
    callback=[early_stop,check_point])
print(results)

# %%
from sklearn.metrics import confusion_matrix,classification_report
y_true=np.concatenate([y for x,y in test_ds], axis=0)
y_pred_prob=model.predict(test_ds)
y_pred=(y_pred_prob>0.5).astype(int).flatten()
print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred,target_names=["real","fake"]))

# %%
false_positive_index=np.where((y_true==0)&(y_pred==1))[0][:5]
false_negative_index=np.where((y_true==1)&(y_pred==0))[0][:5]
test_filepaths = test_ds.file_paths
false_positive_paths = [test_filepaths[i] for i in false_positive_index]
false_negative_paths = [test_filepaths[i] for i in false_negative_index]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, path in enumerate(false_positive_paths):
    img = Image.open(path)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Real → called Fake")
    axes[0, i].axis("off")

for i, path in enumerate(false_negative_paths):
    img = Image.open(path)
    axes[1, i].imshow(img)
    axes[1, i].set_title("Fake → called Real")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()


