import gradio as gr
import cv2
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Sumeet\PycharmProject\SugarcaneDiseasePrediction\DenseNet_Sugarcane_acc90.keras")
labelmappings = {0: 'BacterialBlights',
 1: 'Healthy',
 2: 'Mosaic',
 3: 'RedRot',
 4: 'Rust',
 5: 'Yellow'}


def image_predict(image):
    image = cv2.resize(image,(224,224))
    image = image/255.
    image = image.reshape(1,224,224,3)

    predictions = model.predict(image)
    predictions.round(4)
    label_code = predictions.argmax()
    confidence = round((predictions.tolist())[0][label_code],4)

    return labelmappings[label_code],f"{confidence *100}%"

gr.Blocks(theme=gr.themes.Soft())

demo = gr.Interface(
    fn=image_predict,
    inputs=gr.Image(label="Sugarcane Leaf Image", height=550),
    outputs=[gr.Label(label="Disease Category"),gr.Label(label="Confidence")]
)

demo.launch()
