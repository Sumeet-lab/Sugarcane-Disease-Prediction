import keras
model = keras.models.load_model(r"C:\Sumeet\PycharmProject\SugarcaneDiseasePrediction\DenseNet_Sugarcane_acc90.keras")
print(model.summary())