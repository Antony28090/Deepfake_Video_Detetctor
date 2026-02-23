from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from processor import extract_and_predict_faces

UPLOAD_FOLDER = 'Uploaded_Files'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Ensure upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        try:
            print(f"Processing video: {save_path}")
            result = extract_and_predict_faces(save_path)
            print(f"Result: {result}")
            return render_template('result.html', result=result)
        finally:
            # Clean up the uploaded file after processing
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"Deleted temporary file: {save_path}")
                except Exception as e:
                    print(f"Error deleting file {save_path}: {e}")

    return "Invalid file format"

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)