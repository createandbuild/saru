import os
from os.path import join, dirname, realpath
import flask
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from cv import adversarial

UPLOAD_FOLDER = 'uploads/'
EXPORT_FOLDER = 'cv/data/exports'
UPLOADS_PATH = join(dirname(realpath(__file__)), 'uploads/..')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app=flask.Flask(__name__, template_folder='templates/', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER
app.config["DEBUG"] = True


@app.route('/', methods=['GET', 'POST'])
def home(filename=None, image=None, caption=None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)

        adversarial.attack(full_filename)
        defended_image=os.path.join(app.config['EXPORT_FOLDER'], 'defended.jpg')
        clean_image=os.path.join(app.config['EXPORT_FOLDER'], 'out_clean.jpg')
        adv_image=os.path.join(app.config['EXPORT_FOLDER'], 'out_adv.jpg')

        caption = "JIGGLY PUFF!"

        return render_template('index.html', filename=full_filename,
                image=defended_image, caption=caption)

    elif request.method == 'GET':
        return render_template('index.html', filename=filename, image=defended_image, caption=caption)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/cv', methods=['POST'])
def image_text_summary():
    return


@app.route('/javascript/<path:path>')
def send_js(path):
    return send_from_directory('javascript', path)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()
