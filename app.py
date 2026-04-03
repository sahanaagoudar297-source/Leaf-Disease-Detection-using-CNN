from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import json

# FIXED IMPORTS - No ndimage conflict
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import img_to_array, load_img
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not found. Install: pip install tensorflow")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'leaf-disease-detection-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model SAFELY
model = None
class_names = []
try:
    if TF_AVAILABLE:
        model = load_model('model.h5')
        with open('class_names.json', 'r') as f:
            class_names_dict = json.load(f)
        class_names = [class_names_dict[str(i)] for i in sorted(class_names_dict.keys(), key=int)]
        print(f"✅ Loaded {len(class_names)} classes")
except Exception as e:
    print(f"⚠️ Train model first: python train_model.py ({e})")


RECOVERY_TIPS = {
    'healthy': '✅ Leaf is healthy. Maintain regular watering, sunlight and nutrients.',
    'unhealthy': '❌ Leaf is unhealthy. Remove affected leaves and apply proper treatment.'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        if User.query.filter_by(email=email).first():
            flash('❌ Email exists!')
            return render_template('register.html')
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('✅ Registered! Login now.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('❌ Wrong credentials!')
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model:
        return jsonify({'error': '❌ Train model first!'}), 400

    if 'file' not in request.files:
        return jsonify({'error': '❌ No file uploaded!'}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': '❌ Invalid image!'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = np.argmax(predictions)
        full_class = class_names[predicted_idx]        # e.g. "Apple___Apple_scab"
        confidence = float(predictions[predicted_idx] * 100)

        # ---- Map all classes to Healthy / Unhealthy ----
        full_lower = full_class.lower()
        if "healthy" in full_lower:
            health_label = "Healthy"
            tip = RECOVERY_TIPS['healthy']
        else:
            health_label = "Unhealthy"
            tip = RECOVERY_TIPS['unhealthy']

        # Nicely formatted disease name for display
        pretty_disease = full_class.replace("___", " - ").replace("_", " ")

        return jsonify({
            'health_status': health_label,             # Healthy / Unhealthy
            'disease_name': pretty_disease,           # full class name
            'confidence': f'{confidence:.1f}%',
            'tip': tip,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': f'❌ Processing error: {str(e)}'}), 400


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
