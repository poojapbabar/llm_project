from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from groq import Groq
import os
import pdfplumber
from gtts import gTTS
import speech_recognition as sr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set your Groq API key (ensure you secure this in production)
groq_api_key = os.getenv("GROQ_API_KEY", "enter the api key")
embeddings = Groq(api_key=groq_api_key)

# In-memory storage for chat history
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form.get('user_input')
    file = request.files.get('file')
    language = request.form.get('language')
    text_from_file = ""
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text_from_file += page.extract_text() + "\n"
    
    combined_input = user_input
    if text_from_file:
        combined_input += "\n\n" + text_from_file
    
    try:
        chat_completion = embeddings.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": combined_input,
                }
            ],
            model="llama3-8b-8192",
        )
        output = chat_completion.choices[0].message.content.strip()
        
        # Save user input in chat history
        chat_history.append(user_input)
        
        # Convert output to speech based on selected language
        lang_map = {
            "1": "en",  # English
            "2": "kn",  # Kannada
            "3": "hi"   # Hindi
        }
        tts = gTTS(text=output, lang=lang_map.get(language, "en"), slow=False)
        audio_filename = "output.mp3"
        audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        tts.save(audio_filepath)
        
        return jsonify({'output': output, 'audio_url': f'/uploads/{audio_filename}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/transcribe_voice', methods=['POST'])
def transcribe_voice():
    file = request.files.get('voice')
    
    if not file:
        return jsonify({"error": "No voice file provided"}), 400
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    
    try:
        transcription = recognizer.recognize_google(audio)
        return jsonify({"transcription": transcription})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results; {e}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
