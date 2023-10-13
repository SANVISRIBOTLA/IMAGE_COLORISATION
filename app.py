from flask import Flask, render_template, request,send_from_directory,redirect,url_for,send_file
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
import io
app = Flask(__name__)

app._static_folder = "static"



DIR = r"C:\Users\Dell pc\colorize"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")
@app.route("/")
def index():
    return render_template("demo.html")

@app.route('/page1')
def page1():
    return render_template("colorisation.html")

@app.route('/page2')
def page2():
    return render_template("bw.html")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route("/colorize", methods=["POST"])
def colorize():
    # Get the uploaded image file
    image = request.files["image"]

   

    print("Load model")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    #image = cv2.imread(args["image"])
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # Save the colorized image to a temporary file
    temp_colorized_image_path = 'temp_colorized_image.jpeg'
    cv2.imwrite(temp_colorized_image_path, colorized)

 
# Encode the input image as base64
     # Encode the input image as base64
    _, buffer_input = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer_input).decode("utf-8")

    # Encode the colorized image as base64
    _, buffer_colorized = cv2.imencode(".jpg", colorized)
    base64_colorized_image = base64.b64encode(buffer_colorized).decode("utf-8")

    # Return both images for display
    return render_template("colorisation.html", base64_image=base64_image, base64_colorized_image=base64_colorized_image)



@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "No file part"

    image = request.files["image"]

    if image.filename == "":
        return "No selected file"

    img = Image.open(image)
    img_data = img.getdata()
    lst = []

    for i in img_data:
        lst.append(i[0] * 0.2125 + i[1] * 0.7174 + i[2] * 0.0721)

    new_img = Image.new("L", img.size)
    new_img.putdata(lst)

    # Save the input image to a bytes buffer as PNG
    input_buffer = io.BytesIO()
    img.save(input_buffer, format="PNG")
    input_buffer.seek(0)

    input_image = base64.b64encode(input_buffer.read()).decode("utf-8")

    # Save the processed image to a bytes buffer as PNG
    output_buffer = io.BytesIO()
    new_img.save('temp_colorized_image.jpeg')
    new_img.save(output_buffer, format="PNG")
  

    output_buffer.seek(0)
    processed_image_data = base64.b64encode(output_buffer.read()).decode("utf-8")
    
    return render_template("bw.html", base64_image=processed_image_data, input_image=input_image)

@app.route('/download_colorized_image')
def download_colorized_image():
    # Generate the colorized image (or use a saved image)
    # ...
    # Return the image as a downloadable file
    return send_file('temp_colorized_image.jpeg', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
