from flask import Flask, request, jsonify, send_from_directory
from AWS_AI_Storyteller import generate_text_from_bedrock, text_to_mp3


app = Flask(__name__, static_folder='.')

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    try:
        story = generate_text_from_bedrock(prompt)
        if story:
            mp3_filename = 'harry_potter_story.mp3'
            text_to_mp3(story, mp3_filename)
            return jsonify({'story': story, 'mp3': mp3_filename})
        else:
            print('generate_text_from_bedrock returned None')
            return jsonify({'error': 'Failed to generate story'}), 500
    except Exception as e:
        print(f'Error in /generate-story: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 