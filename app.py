from flask import Flask, render_template, request, jsonify, session, send_file
import os
import json
import markdown
import time
import threading
import uuid
import pickle
from werkzeug.utils import secure_filename
from generatereport import ConsultantReportGenerator

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'static/reports'
app.config['REPORTS_DATA'] = 'static/reports/reports_data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_DATA'], exist_ok=True)

# Dictionary to store background tasks
background_tasks = {}

# Load any existing saved report data
def load_report_data():
    try:
        # Load saved report data
        for file in os.listdir(app.config['REPORTS_DATA']):
            if file.endswith('.pkl'):
                task_id = file.split('.')[0]
                file_path = os.path.join(app.config['REPORTS_DATA'], file)
                with open(file_path, 'rb') as f:
                    report_data = pickle.load(f)
                    background_tasks[task_id] = report_data
        print(f"Loaded {len(background_tasks)} saved reports")
    except Exception as e:
        print(f"Error loading saved reports: {str(e)}")

# Save report data to persistent storage
def save_report_data(task_id):
    try:
        if task_id in background_tasks and background_tasks[task_id]['status'] == 'completed':
            file_path = os.path.join(app.config['REPORTS_DATA'], f"{task_id}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(background_tasks[task_id], f)
            print(f"Saved report data for task {task_id}")
    except Exception as e:
        print(f"Error saving report data: {str(e)}")

# Load existing reports on startup
load_report_data()

# Progress callback to update task status
def update_progress(task_id, message, progress, current_step=None, total_steps=None, search_results=None):
    if task_id in background_tasks:
        # Make sure progress doesn't exceed 99% until we're completely done
        if progress >= 100 and message != "Report complete!" and message != "Response complete!":
            progress = 99
            
        progress_data = {
            'percent': progress,
            'message': message
        }
        
        if current_step is not None and total_steps is not None:
            progress_data['current_step'] = current_step
            progress_data['total_steps'] = total_steps
            
        background_tasks[task_id]['progress'] = progress_data
        
        # Add search results if provided
        if search_results:
            background_tasks[task_id]['search_results'] = search_results

def generate_report_task(query, task_id, mode="concise"):
    """Background task to generate the consultant report"""
    try:
        # Create progress callback
        def progress_callback(message, progress, search_results=None):
            current_step = background_tasks[task_id].get('current_step', 1)
            if 'Generating' in message:
                current_step += 1
            
            # Adjust total steps based on mode
            total_steps = 7 if mode == "detailed" else 5  # Concise mode has fewer steps
            
            background_tasks[task_id]['current_step'] = current_step
            update_progress(
                task_id, 
                message, 
                progress, 
                current_step, 
                total_steps,
                search_results
            )
        
        # Initialize task with starting step
        background_tasks[task_id]['current_step'] = 1
        
        # Create generator with progress callback and mode
        generator = ConsultantReportGenerator(progress_callback, mode)
        output_file = generator.generate_consultant_report(query)
        
        # Read the markdown content
        with open(output_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code']
        )
        
        # Initialize chat history
        chat_history = [
            {"role": "system", "content": "You are an AI research assistant helping users explore topics from a comprehensive report."},
            {"role": "assistant", "content": f"I've generated a {'comprehensive report' if mode == 'detailed' else 'concise response'} on '{query}'. You can ask me any questions about the {'report' if mode == 'detailed' else 'research'} or request further analysis on specific aspects."}
        ]
        
        # Save the result
        background_tasks[task_id]['status'] = 'completed'
        background_tasks[task_id]['result'] = {
            'html': html_content,
            'raw_markdown': markdown_content,
            'filename': output_file,
            'title': query,
            'mode': mode
        }
        background_tasks[task_id]['chat_history'] = chat_history
        
        # Make sure progress bar reaches 100%
        total_steps = 7 if mode == "detailed" else 5
        update_progress(
            task_id, 
            "Report complete!" if mode == "detailed" else "Response complete!", 
            100, total_steps, total_steps, None
        )
        
        # Save report data to persistent storage
        save_report_data(task_id)
        
    except Exception as e:
        background_tasks[task_id]['status'] = 'failed'
        background_tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Get the selected mode (default to concise if not provided)
    mode = request.form.get('mode', 'concise')
    if mode not in ['concise', 'detailed']:
        mode = 'concise'  # Default to concise if invalid mode
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    background_tasks[task_id] = {
        'status': 'running',
        'created_at': time.time(),
        'query': query,
        'mode': mode,
        'progress': {
            'percent': 0,
            'message': 'Initializing...',
            'current_step': 1,
            'total_steps': 7 if mode == 'detailed' else 5
        },
        'search_results': []
    }
    
    # Start background task
    thread = threading.Thread(target=generate_report_task, args=(query, task_id, mode))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def task_status(task_id):
    if task_id not in background_tasks:
        return jsonify({'status': 'not_found'}), 404
    
    task = background_tasks[task_id]
    response = {'status': task['status']}
    
    if task['status'] == 'completed':
        response['result'] = {
            'title': task['result']['title'],
            'filename': task['result']['filename'],
            'mode': task['result']['mode']
        }
    elif task['status'] == 'failed':
        response['error'] = task.get('error', 'An unknown error occurred')
    
    # Add progress information
    if 'progress' in task:
        response['progress'] = task['progress']
    
    # Add search results if available
    if 'search_results' in task:
        response['search_results'] = task['search_results']
        
    return jsonify(response)

# New endpoint to get report content without loading a new page
@app.route('/report_content/<task_id>')
def report_content(task_id):
    if task_id not in background_tasks or background_tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Report not found or not completed yet'}), 404
    
    result = background_tasks[task_id]['result']
    # Include chat history
    chat_history = background_tasks[task_id].get('chat_history', [])
    # Filter out system messages for the client
    client_chat_history = [msg for msg in chat_history if msg['role'] != 'system']
    
    return jsonify({
        'html_content': result['html'],
        'title': result['title'],
        'mode': result.get('mode', 'detailed'),  # Default to detailed for backward compatibility
        'chat_history': client_chat_history
    })

# Keep this endpoint for backward compatibility and direct downloads
@app.route('/report/<task_id>')
def show_report(task_id):
    if task_id not in background_tasks or background_tasks[task_id]['status'] != 'completed':
        return render_template('error.html', message="Report not found or not completed yet")
    
    result = background_tasks[task_id]['result']
    return render_template('report.html', 
                          title=result['title'],
                          html_content=result['html'],
                          task_id=task_id)

@app.route('/download/<task_id>')
def download_report(task_id):
    if task_id not in background_tasks or background_tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Report not found or not completed yet'}), 404
    
    # Return the markdown file for download
    filename = background_tasks[task_id]['result']['filename']
    return send_file(filename, as_attachment=True)

# New endpoint to handle chat messages
@app.route('/chat/<task_id>', methods=['POST'])
def chat(task_id):
    if task_id not in background_tasks or background_tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Report not found or not completed yet'}), 404
    
    # Get the user message
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    
    # Get report content to use as context
    report_content = background_tasks[task_id]['result']['raw_markdown']
    
    # Get existing chat history
    if 'chat_history' not in background_tasks[task_id]:
        mode = background_tasks[task_id]['result'].get('mode', 'detailed')
        background_tasks[task_id]['chat_history'] = [
            {"role": "system", "content": "You are an AI research assistant helping users explore topics from a comprehensive report."},
            {"role": "assistant", "content": f"I've generated a {'comprehensive report' if mode == 'detailed' else 'concise response'} on '{background_tasks[task_id]['query']}'. You can ask me any questions about the {'report' if mode == 'detailed' else 'research'} or request further analysis on specific aspects."}
        ]
    
    chat_history = background_tasks[task_id]['chat_history']
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_message})
    
    try:
        # Call the OpenRouter API to get a response
        import requests
        
        # OpenRouter API endpoint
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Create the messages array with system message, context, and chat history
        messages = [
            {
                "role": "system",
                "content": f"""You are an AI research assistant helping users explore topics from a comprehensive report. Do not limit yourself to this report, this is just to proide ou some helpful context. If there is something you don't know, you can refer this report.  
                The following is the content of a research report that has been generated:
                
                {report_content[:50000]}  # Limit context to avoid token limits
                
                You can use the report above as a context but do not limit yourself to the existing context, use your own knowledge wherever required. You can try to Answer the user's questions based on this report.
                
                Be helpful and accurate. You know that you are smart. 
                """
            }
        ]
        
        # Add chat history (except for the system message which is replaced with our new context-rich one)
        for msg in chat_history:
            if msg["role"] != "system":
                messages.append(msg)
        
        payload = {
            "model": "google/gemini-2.0-flash-001",  # Use the same model as report generation
            "messages": messages
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                return jsonify({'error': f"API Error: {result['error']}"}), 500
                
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Add assistant response to chat history
            chat_history.append({"role": "assistant", "content": assistant_response})
            
            # Save updated chat history
            background_tasks[task_id]['chat_history'] = chat_history
            save_report_data(task_id)
            
            return jsonify({
                'response': assistant_response,
                'chat_history': [msg for msg in chat_history if msg['role'] != 'system']  # Filter out system message
            })
        else:
            return jsonify({'error': f"HTTP Error: {response.status_code} - {response.text}"}), 500
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'error': f"Failed to get response: {str(e)}"}), 500

# Add custom route to track search results
@app.route('/add_search_result/<task_id>', methods=['POST'])
def add_search_result(task_id):
    if task_id not in background_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing url in request'}), 400
    
    # Add to search results
    if 'search_results' not in background_tasks[task_id]:
        background_tasks[task_id]['search_results'] = []
    
    background_tasks[task_id]['search_results'].append({
        'url': data.get('url'),
        'title': data.get('title', 'Source'),
        'source': data.get('source', '')
    })
    
    return jsonify({'success': True})

# Route to clean up old tasks (can be called periodically)
@app.route('/cleanup', methods=['POST'])
def cleanup_tasks():
    # Only allow with admin key
    if request.headers.get('X-Admin-Key') != os.getenv('ADMIN_KEY'):
        return jsonify({'error': 'Unauthorized'}), 403
        
    current_time = time.time()
    expired_tasks = []
    
    # Find tasks older than 24 hours
    for task_id, task in background_tasks.items():
        if current_time - task['created_at'] > 86400:  # 24 hours
            expired_tasks.append(task_id)
            
            # Also remove the saved file
            try:
                file_path = os.path.join(app.config['REPORTS_DATA'], f"{task_id}.pkl")
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing saved report: {str(e)}")
    
    # Remove expired tasks
    for task_id in expired_tasks:
        del background_tasks[task_id]
    
    return jsonify({'removed': len(expired_tasks)})

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """API endpoint to get list of all reports"""
    reports = []
    
    # Convert background_tasks to a list of report summaries
    for task_id, task in background_tasks.items():
        if task['status'] == 'completed':
            reports.append({
                'id': task_id,
                'title': task['result'].get('title', ''),
                'created_at': task['created_at'],
                'mode': task['result'].get('mode', 'detailed')  # Include mode in report summary
            })
    
    # Sort by creation date (newest first)
    reports.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify(reports)

if __name__ == '__main__':
    app.run(debug=True)
