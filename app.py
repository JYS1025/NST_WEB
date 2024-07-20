from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import nst

app = Flask(__name__)

# 업로드된 파일을 저장할 디렉토리 설정
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용되는 파일 확장자 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 파일 업로드 처리
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        content_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(content_img_path)
        
        # 여기서 PyTorch 모델을 사용하여 Neural Style Transfer 수행
        target_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'target.jpg')  # 타겟 이미지 경로
        output_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')  # 결과 이미지 경로
        nst(content_img_path, target_img_path, output_img_path)
        
        return redirect(url_for('result', filename='output.jpg'))
    return redirect(url_for('index'))

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
