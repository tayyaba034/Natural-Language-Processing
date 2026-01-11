from flask import Flask, render_template, request, jsonify
from summarizer import summarize_all

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'extractive': '', 'abstractive': ''})
    
    summaries = summarize_all(text)
    return jsonify(summaries)

if __name__ == '__main__':
    app.run(debug=True)
