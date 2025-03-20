from flask import Flask, render_template, request, jsonify, session, send_file
import os
import json
import markdown
import time
import threading
import uuid
from werkzeug.utils import secure_filename
from generatereport import ConsultantReportGenerator

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'static/reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary to store background tasks
background_tasks = {}

# Progress callback to update task status
def update_progress(task_id, message, progress, current_step=None, total_steps=None, search_results=None):
    if task_id in background_tasks:
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

def generate_report_task(query, task_id):
    """Background task to generate the consultant report"""
    try:
        # Create progress callback
        def progress_callback(message, progress, search_results=None):
            current_step = background_tasks[task_id].get('current_step', 1)
            if 'Generating' in message:
                current_step += 1
            
            background_tasks[task_id]['current_step'] = current_step
            update_progress(
                task_id, 
                message, 
                progress, 
                current_step, 
                7,  # Total steps
                search_results
            )
        
        # Initialize task with starting step
        background_tasks[task_id]['current_step'] = 1
        
        # Create generator with progress callback
        generator = ConsultantReportGenerator(progress_callback)
        output_file = generator.generate_consultant_report(query)
        
        # Read the markdown content
        with open(output_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code']
        )
        
        # Save the result
        background_tasks[task_id]['status'] = 'completed'
        background_tasks[task_id]['result'] = {
            'html': html_content,
            'raw_markdown': markdown_content,
            'filename': output_file,
            'title': query
        }
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
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    background_tasks[task_id] = {
        'status': 'running',
        'created_at': time.time(),
        'query': query,
        'progress': {
            'percent': 0,
            'message': 'Initializing...',
            'current_step': 1,
            'total_steps': 7
        },
        'search_results': []
    }
    
    # Start background task
    thread = threading.Thread(target=generate_report_task, args=(query, task_id))
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
            'filename': task['result']['filename']
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
    return jsonify({
        'html_content': result['html'],
        'title': result['title']
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
                'created_at': task['created_at']
            })
    
    # Sort by creation date (newest first)
    reports.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify(reports)
if __name__ == '__main__':
    app.run(debug=True)
