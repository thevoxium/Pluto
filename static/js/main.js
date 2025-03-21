document.addEventListener('DOMContentLoaded', function() {
    const reportForm = document.getElementById('report-form');
    const researchOverlay = document.getElementById('research-overlay');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const progressIndicator = document.getElementById('progress-indicator');
    const currentQuery = document.getElementById('current-query');
    const searchResults = document.getElementById('search-results');
    const elapsedTime = document.getElementById('elapsed-time');
    const reportSection = document.getElementById('report-section');
    const reportContent = document.getElementById('report-content');
    const downloadReportBtn = document.getElementById('download-report');
    const newResearchBtn = document.getElementById('new-research-btn');
    const modeSelector = document.getElementById('mode-selector');
    const modeBadge = document.getElementById('mode-badge');
    const statusSteps = document.getElementById('status-steps');
    
    // Status steps
    const iconThinking = document.getElementById('icon-thinking');
    const iconResearch = document.getElementById('icon-research');
    const iconAnalysis = document.getElementById('icon-analysis');
   
    // Report history elements
    const reportHistoryContainer = document.getElementById('report-history');
    const emptyReportsState = document.getElementById('empty-reports-state');
    const reportModal = document.getElementById('report-modal');
    const modalReportTitle = document.getElementById('modal-report-title');
    const modalReportContent = document.getElementById('modal-report-content');
    const modalDownloadBtn = document.getElementById('modal-download-report');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    
    // Chat elements
    const chatContainer = document.getElementById('chat-container');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendMessageBtn = document.getElementById('send-message-btn');
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    
    // Modal chat elements
    const modalChatContainer = document.getElementById('modal-chat-container');
    const modalChatMessages = document.getElementById('modal-chat-messages');
    const modalChatToggleBtn = document.getElementById('modal-chat-toggle-btn');
    
    // Timer variables
    let startTime;
    let timerInterval;
    let currentTaskId = null;
    
    // Track current mode
    let currentMode = 'concise';
    
    // Chat state
    let isWaitingForResponse = false;
    
    // Load report history on page load
    loadReportHistory();
    
    // Get reference to the scroll button
    const scrollButton = document.getElementById('scroll-button');
    
    // Variable to track if we're at the bottom
    let isAtBottom = false;
    
    // Mode selector handler
    if (modeSelector) {
        modeSelector.addEventListener('change', function() {
            currentMode = this.value;
            updateModeIndicator(currentMode);
        });
    }
    
    // Initialize mode indicator
    updateModeIndicator(modeSelector ? modeSelector.value : 'concise');
    
    // Function to update the mode indicator
    function updateModeIndicator(mode) {
        if (!modeBadge) return;
        
        // Update badge text
        const modeText = modeBadge.querySelector('span');
        if (modeText) {
            modeText.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
        }
        
        // Update badge icon and style
        const modeIcon = modeBadge.querySelector('i');
        if (modeIcon) {
            if (mode === 'concise') {
                modeIcon.className = 'fas fa-bolt';
                modeBadge.className = 'mode-badge mode-concise';
            } else {
                modeIcon.className = 'fas fa-book';
                modeBadge.className = 'mode-badge mode-detailed';
            }
        }
        
        // Update status steps display based on mode
        if (statusSteps) {
            if (mode === 'concise') {
                // Simplified steps for concise mode
                statusSteps.classList.add('concise-mode');
            } else {
                statusSteps.classList.remove('concise-mode');
            }
        }
    }
    
    // Function to update button state based on scroll position
    function updateScrollButton() {
        if (!reportSection || reportSection.classList.contains('hidden')) return;
        
        // Get positions
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollPosition = window.scrollY;
        
        // Check if near bottom (within 300px of bottom)
        const isNearBottom = scrollPosition + windowHeight > documentHeight - 300;
        
        if (isNearBottom) {
            // Change to "Scroll to top"
            scrollButton.classList.add('scrolling-up');
            scrollButton.querySelector('.scroll-text').textContent = 'Report';
            scrollButton.querySelector('i').className = 'fas fa-chevron-up';
            isAtBottom = true;
        } else {
            // Change to "Scroll to chat"
            scrollButton.classList.remove('scrolling-up');
            scrollButton.querySelector('.scroll-text').textContent = 'Chat';
            scrollButton.querySelector('i').className = 'fas fa-chevron-down';
            isAtBottom = false;
        }
    }
    
    // Add click event listener to the scroll button
    if (scrollButton) {
        scrollButton.addEventListener('click', function() {
            if (isAtBottom) {
                // Scroll to the top of the report
                reportContent.scrollIntoView({ behavior: 'smooth' });
                isAtBottom = false;
            } else {
                // Scroll to the chat container
                if (chatContainer) {
                    chatContainer.scrollIntoView({ behavior: 'smooth' });
                    isAtBottom = true;
                    
                    // If chat is collapsed, expand it
                    if (chatContainer.classList.contains('collapsed') && chatToggleBtn) {
                        toggleChatContainer(chatContainer, chatToggleBtn);
                    }
                    
                    // Focus on input after scrolling
                    setTimeout(() => {
                        if (chatInput) chatInput.focus();
                    }, 800);
                }
            }
        });
    }
    
    // Function to show scroll button when report is ready
    function showScrollButtonForReport() {
        if (!scrollButton) return;
        
        // Make button visible
        scrollButton.classList.remove('scroll-hidden');
        
        // Start listening for scroll events
        window.addEventListener('scroll', updateScrollButton);
        
        // Trigger once to set initial state
        updateScrollButton();
    }
    
    // Function to hide scroll button
    function hideScrollButton() {
        if (!scrollButton) return;
        
        scrollButton.classList.add('scroll-hidden');
        
        // Stop listening for scroll events
        window.removeEventListener('scroll', updateScrollButton);
    }
    
    // Auto-resize textarea as user types
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Toggle chat container
    if (chatToggleBtn) {
        chatToggleBtn.addEventListener('click', function() {
            toggleChatContainer(chatContainer, chatToggleBtn);
        });
    }
    
    // Toggle modal chat container
    if (modalChatToggleBtn) {
        modalChatToggleBtn.addEventListener('click', function() {
            toggleChatContainer(modalChatContainer, modalChatToggleBtn);
        });
    }
    
    // Send message when enter key is pressed (but allow shift+enter for new lines)
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Send message when send button is clicked
    if (sendMessageBtn) {
        sendMessageBtn.addEventListener('click', sendMessage);
    }
    
    if (reportForm) {
        reportForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value.trim();
            if (!query) {
                showNotification('Please enter a research query', 'error');
                return;
            }
            
            // Get the selected mode
            currentMode = modeSelector ? modeSelector.value : 'concise';
            
            // Show loading state on button
            const generateBtn = document.getElementById('generate-btn');
            const originalBtnText = generateBtn.innerHTML;
            generateBtn.innerHTML = '<span class="spinner"></span> Initializing...';
            generateBtn.disabled = true;
            
            // Show research overlay with animation
            researchOverlay.style.display = 'block';
            researchOverlay.style.opacity = '0';
            setTimeout(() => {
                researchOverlay.style.opacity = '1';
                researchOverlay.style.transition = 'opacity 0.3s ease';
            }, 10);
            
            progressBar.style.width = '5%';
            statusMessage.textContent = 'Generating search queries...';
            
            // Update the progress indicator based on mode
            progressIndicator.textContent = currentMode === 'detailed' ? '1/7' : '1/5';
            
            currentQuery.textContent = query;
            
            // Update mode indicator in the research view
            updateModeIndicator(currentMode);
            
            // Reset research UI
            resetResearchUI();
            
            // Start timer
            startTimer();
            
            // Set initial status
            setStatus('thinking');
            
            // Submit form data with mode
            const formData = new FormData(reportForm);
            formData.append('mode', currentMode); // Ensure mode is included
            
            fetch('/generate', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                generateBtn.innerHTML = originalBtnText;
                generateBtn.disabled = false;
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store task ID for download link
                currentTaskId = data.task_id;
                
                // Update download link
                downloadReportBtn.href = `/download/${currentTaskId}`;
                
                // Start polling for status
                pollTaskStatus(currentTaskId);
            })
            .catch(error => {
                // Reset button state
                generateBtn.innerHTML = originalBtnText;
                generateBtn.disabled = false;
                
                console.error('Error:', error);
                showError('Failed to generate report: ' + error.message);
            });
        });
    }
    
    if (newResearchBtn) {
        newResearchBtn.addEventListener('click', function() {
            // Fade out animation
            researchOverlay.style.opacity = '0';
            setTimeout(() => {
                researchOverlay.style.display = 'none';
            }, 300);
            
            stopTimer();
            // Reset form if needed
            document.getElementById('query').value = '';
            
            // Hide scroll button
            hideScrollButton();
        });
    }
    
    // Modal close button
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', function() {
            closeReportModal();
        });
        
        // Also close when clicking outside the modal content
        if (reportModal) {
            reportModal.addEventListener('click', function(e) {
                if (e.target === reportModal) {
                    closeReportModal();
                }
            });
        }
        
        // Close on ESC key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && reportModal && reportModal.classList.contains('open')) {
                closeReportModal();
            }
        });
    }
    
    // Function to toggle chat container visibility
    function toggleChatContainer(container, toggleBtn) {
        const isCollapsed = container.classList.contains('collapsed');
        
        if (isCollapsed) {
            // Expand
            container.classList.remove('collapsed');
            container.classList.add('opening');
            toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
            setTimeout(() => {
                container.classList.remove('opening');
            }, 300);
        } else {
            // Collapse
            container.classList.add('closing');
            toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
            setTimeout(() => {
                container.classList.add('collapsed');
                container.classList.remove('closing');
            }, 300);
        }
    }
    
    // Function to send a message to the chat
    function sendMessage() {
        if (!chatInput || !currentTaskId || isWaitingForResponse) return;
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Disable input and button while processing
        chatInput.disabled = true;
        if (sendMessageBtn) sendMessageBtn.disabled = true;
        isWaitingForResponse = true;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        // Add thinking indicator
        const thinkingIndicator = document.createElement('div');
        thinkingIndicator.className = 'thinking-indicator';
        thinkingIndicator.innerHTML = `
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        `;
        chatMessages.appendChild(thinkingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send message to server
        fetch(`/chat/${currentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove thinking indicator
            if (thinkingIndicator) {
                chatMessages.removeChild(thinkingIndicator);
            }
            
            if (data.error) {
                showNotification(`Error: ${data.error}`, 'error');
                return;
            }
            
            // Add assistant's response to chat
            addMessageToChat('assistant', data.response);
            
            // Enable input and button
            chatInput.disabled = false;
            if (sendMessageBtn) sendMessageBtn.disabled = false;
            chatInput.focus();
            isWaitingForResponse = false;
        })
        .catch(error => {
            console.error('Error sending message:', error);
            
            // Remove thinking indicator
            if (thinkingIndicator) {
                chatMessages.removeChild(thinkingIndicator);
            }
            
            // Show error in chat
            addMessageToChat('assistant', `Sorry, there was an error processing your request: ${error.message}`);
            
            // Enable input and button
            chatInput.disabled = false;
            if (sendMessageBtn) sendMessageBtn.disabled = false;
            isWaitingForResponse = false;
            
            showNotification('Failed to send message. Please try again.', 'error');
        });
    }
    
    // Function to add a message to the chat
    function addMessageToChat(role, content, timestamp) {
        if (!chatMessages) return;
        
        const messageTime = timestamp || new Date();
        const formattedTime = messageTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = `message-bubble ${role}-bubble`;
        
        // Convert markdown-like syntax to HTML
        let formattedContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        bubbleDiv.innerHTML = formattedContent;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = formattedTime;
        
        bubbleDiv.appendChild(timeSpan);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to load chat history from server response
    function loadChatHistory(history) {
        if (!chatMessages || !history || !Array.isArray(history)) return;
        
        // Clear existing messages
        chatMessages.innerHTML = '';
        
        // Add each message
        history.forEach(msg => {
            if (msg.role !== 'system') {
                addMessageToChat(msg.role, msg.content);
            }
        });
    }
    
    function pollTaskStatus(taskId) {
        // Track consecutive errors
        let errorCount = 0;
        
        // Store for adding to history later
        let reportTitle = "";
        let reportMode = "concise";
        
        const pollInterval = setInterval(() => {
            fetch(`/status/${taskId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Reset error count on successful response
                    errorCount = 0;
                    
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        stopTimer();
                        
                        // Complete all status steps
                        setStatus('completed');
                        
                        // Update progress bar to 100%
                        progressBar.style.width = '100%';
                        statusMessage.textContent = data.result.mode === 'detailed' ? 'Report complete!' : 'Response complete!';
                        
                        // Store the mode
                        reportMode = data.result.mode || 'concise';
                        
                        // When completed, fetch the report content and show it in the same page
                        fetch(`/report_content/${taskId}`)
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error! Status: ${response.status}`);
                                }
                                return response.json();
                            })
                            .then(reportData => {
                                if (reportData.error) {
                                    throw new Error(reportData.error);
                                }
                                
                                // Save report title for history
                                reportTitle = reportData.title;
                                reportMode = reportData.mode || 'concise';
                                
                                // Show report section with animation
                                reportSection.classList.remove('hidden');
                                reportSection.style.opacity = '0';
                                reportSection.style.transform = 'translateY(20px)';
                                
                                setTimeout(() => {
                                    reportSection.style.opacity = '1';
                                    reportSection.style.transform = 'translateY(0)';
                                    reportSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                                }, 10);
                                
                                // Set report content
                                reportContent.innerHTML = reportData.html_content;
                                
                                // Load chat history if available
                                if (reportData.chat_history && reportData.chat_history.length > 0) {
                                    loadChatHistory(reportData.chat_history);
                                }
                                
                                // Apply syntax highlighting to code blocks
                                document.querySelectorAll('pre code').forEach((block) => {
                                    hljs.highlightBlock(block);
                                });
                                
                                // Make tables responsive
                                document.querySelectorAll('table').forEach((table) => {
                                    if (!table.parentNode.classList.contains('table-responsive')) {
                                        const wrapper = document.createElement('div');
                                        wrapper.className = 'table-responsive';
                                        table.parentNode.insertBefore(wrapper, table);
                                        wrapper.appendChild(table);
                                    }
                                });
                                
                                // Add to report history
                                const previewText = extractPreviewFromHtml(reportData.html_content);
                                addReportToHistory(taskId, reportTitle, previewText, reportMode);
                                
                                // Scroll to report section with smooth animation
                                reportSection.scrollIntoView({ behavior: 'smooth' });
                                
                                // Show scroll button
                                showScrollButtonForReport();
                                
                                // Show completion notification
                                showNotification(`${reportMode === 'detailed' ? 'Report' : 'Response'} generated successfully!`, 'success');
                            })
                            .catch(error => {
                                console.error('Error loading report:', error);
                                showError('Failed to load report: ' + error.message);
                                hideScrollButton();
                            });
                    } else if (data.status === 'failed') {
                        clearInterval(pollInterval);
                        stopTimer();
                        showError('Report generation failed: ' + (data.error || 'Unknown error'));
                        hideScrollButton();
                    } else if (data.status === 'running') {
                        // Update UI with progress info
                        updateResearchUI(data);
                    }
                })
                .catch(error => {
                    console.error('Error polling status:', error);
                    errorCount++;
                    
                    // If we've had 5 consecutive errors, stop polling
                    if (errorCount >= 5) {
                        clearInterval(pollInterval);
                        stopTimer();
                        showError('Failed to get status updates: ' + error.message);
                        hideScrollButton();
                    }
                    // Don't stop polling on occasional network errors
                });
        }, 1000); // Poll every second for responsive UI
    }
    
    function updateResearchUI(data) {
        // Update progress if available
        if (data.progress) {
            // Update progress bar
            let percent = 0;
            if (typeof data.progress === 'object') {
                percent = data.progress.percent || 0;
                
                // Update step indicator
                if (data.progress.current_step && data.progress.total_steps) {
                    progressIndicator.textContent = `${data.progress.current_step}/${data.progress.total_steps}`;
                }
                
                // Update status message
                if (data.progress.message) {
                    statusMessage.textContent = data.progress.message;
                    
                    // Update status icons based on the current phase
                    const totalSteps = data.progress.total_steps || (currentMode === 'detailed' ? 7 : 5);
                    
                    if (data.progress.current_step <= Math.ceil(totalSteps * 0.3)) {
                        setStatus('thinking');
                    } else if (data.progress.current_step <= Math.ceil(totalSteps * 0.7)) {
                        setStatus('research');
                    } else {
                        setStatus('analysis');
                    }
                }
            } else {
                percent = data.progress;
            }
            
            // Ensure the progress bar transitions smoothly
            progressBar.style.transition = 'width 0.5s ease';
            progressBar.style.width = `${percent}%`;
        }
        
        // Update search results if available
        if (data.search_results && data.search_results.length > 0) {
            updateSearchResults(data.search_results);
        }
    }
    
    function updateSearchResults(results) {
        // If first time, create the structure
        if (searchResults.innerHTML === '') {
            // Create header with toggle functionality
            const resultsHeader = document.createElement('h3');
            resultsHeader.innerHTML = 'Sources being analyzed <span class="result-count">(' + results.length + ')</span>';
            
            const resultsContent = document.createElement('div');
            resultsContent.className = 'results-list';
            
            searchResults.appendChild(resultsHeader);
            searchResults.appendChild(resultsContent);
            
            // Add toggle functionality
            resultsHeader.addEventListener('click', function() {
                resultsContent.classList.toggle('collapsed');
                resultsHeader.classList.toggle('collapsed');
            });
        } else {
            // Update count of results
            const resultCount = searchResults.querySelector('.result-count');
            if (resultCount) {
                resultCount.textContent = `(${results.length})`;
            }
        }
        
        // Get the content container
        const resultsContent = searchResults.querySelector('.results-list');
        if (resultsContent) {
            // Clear existing results
            resultsContent.innerHTML = '';
            
            // Add each result
            results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item new'; // Add 'new' class for animation
                
                const resultSource = document.createElement('div');
                resultSource.className = 'result-source';
                resultSource.textContent = result.title || result.source || 'Source';
                
                const resultUrl = document.createElement('div');
                resultUrl.className = 'result-url';
                resultUrl.textContent = result.url || '';
                
                resultItem.appendChild(resultSource);
                resultItem.appendChild(resultUrl);
                resultsContent.appendChild(resultItem);
                
                // Remove the 'new' class after animation completes
                setTimeout(() => {
                    resultItem.classList.remove('new');
                }, 1000);
            });
        }
    }
    
    function resetResearchUI() {
        searchResults.innerHTML = '';
        progressBar.style.width = '0%';
        elapsedTime.textContent = '00:00';
        reportSection.classList.add('hidden');
        reportContent.innerHTML = '';
        
        // Hide scroll button
        hideScrollButton();
        
        // Reset chat
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        // Reset status steps
        const statusSteps = document.querySelectorAll('.status-step');
        statusSteps.forEach(step => {
            step.className = 'status-step';
        });
    }
    
    function setStatus(status) {
        // Get all status steps
        const thinkingStep = document.getElementById('icon-thinking');
        const researchStep = document.getElementById('icon-research');
        const analysisStep = document.getElementById('icon-analysis');
        
        if (!thinkingStep || !researchStep || !analysisStep) return;
        
        // Reset all steps
        thinkingStep.className = 'status-step';
        researchStep.className = 'status-step';
        analysisStep.className = 'status-step';
        
        // Update steps based on current status
        switch(status) {
            case 'thinking':
                thinkingStep.className = 'status-step active';
                break;
                
            case 'research':
                thinkingStep.className = 'status-step completed';
                researchStep.className = 'status-step active';
                break;
                
            case 'analysis':
                thinkingStep.className = 'status-step completed';
                researchStep.className = 'status-step completed';
                analysisStep.className = 'status-step active';
                break;
                
            case 'completed':
                thinkingStep.className = 'status-step completed';
                researchStep.className = 'status-step completed';
                analysisStep.className = 'status-step completed';
                break;
        }
    }
    
    function startTimer() {
        startTime = new Date();
        timerInterval = setInterval(updateTimer, 1000);
    }
    
    function stopTimer() {
        clearInterval(timerInterval);
    }
    
    function updateTimer() {
        const now = new Date();
        const elapsed = Math.floor((now - startTime) / 1000); // seconds
        
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        
        elapsedTime.textContent = `${minutes}:${seconds}`;
    }
    
    function showError(message) {
        // If searchResults is empty, create structure
        if (searchResults.innerHTML === '') {
            searchResults.innerHTML = `<h3>Status</h3>`;
        }
        
        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        
        // Clear any existing content and append error
        searchResults.innerHTML = '';
        searchResults.appendChild(errorDiv);
        
        stopTimer();
    }
    
    // Show notification toast
    function showNotification(message, type = 'info') {
        // Create notification element if it doesn't exist
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.style.position = 'fixed';
            container.style.bottom = '20px';
            container.style.right = '20px';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        // Create notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.backgroundColor = type === 'error' ? 'var(--error-color)' : 
                                            type === 'success' ? 'var(--success-color)' : 
                                            'var(--accent-color)';
        notification.style.color = 'white';
        notification.style.padding = '12px 20px';
        notification.style.borderRadius = 'var(--border-radius-md)';
        notification.style.marginTop = '10px';
        notification.style.boxShadow = 'var(--shadow-md)';
        notification.style.transform = 'translateX(120%)';
        notification.style.transition = 'transform 0.3s ease';
        
        // Add icon based on type
        const icon = type === 'error' ? 'fas fa-exclamation-circle' : 
                    type === 'success' ? 'fas fa-check-circle' : 
                    'fas fa-info-circle';
      notification.innerHTML = `<i class="${icon}" style="margin-right: 10px;"></i> ${message}`;
        
        // Add to container
        const container = document.getElementById('notification-container');
        container.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Remove after a delay
        setTimeout(() => {
            notification.style.transform = 'translateX(120%)';
            setTimeout(() => {
                container.removeChild(notification);
            }, 300);
        }, 5000);
    }
    
    // Print functionality
    if (document.getElementById('print-report')) {
        document.getElementById('print-report').addEventListener('click', function() {
            window.print();
        });
    }
    
    // Function to add a report to history
    function addReportToHistory(taskId, title, previewText, mode = 'concise') {
        if (!reportHistoryContainer) return;
        
        const reports = getReportHistory();
        
        // Create new report object
        const newReport = {
            id: taskId,
            title: title,
            preview: previewText || `Strategic analysis and recommendations for ${title}`,
            date: new Date().toISOString(),
            mode: mode,
            isNew: true
        };
        
        // Add to beginning of array
        reports.unshift(newReport);
        
        // Keep only the most recent 20 reports
        if (reports.length > 20) {
            reports.pop();
        }
        
        // Save to localStorage
        saveReportHistory(reports);
        
        // Refresh the display
        displayReportHistory(reports);
    }
    
    function loadReportHistory() {
        if (!reportHistoryContainer) return;
        
        const reports = getReportHistory();
        displayReportHistory(reports);
    }
    
    function displayReportHistory(reports) {
        // If container doesn't exist, exit
        if (!reportHistoryContainer) return;

        // Clear report container except empty state
        const elements = reportHistoryContainer.querySelectorAll(':not(#empty-reports-state)');
        elements.forEach(el => el.remove());
        
        // Show or hide empty state
        if (reports.length === 0) {
            if (emptyReportsState) emptyReportsState.style.display = 'flex';
            return;
        } else {
            if (emptyReportsState) emptyReportsState.style.display = 'none';
        }
        
        // Add each report card
        reports.forEach(report => {
            const card = createReportCard(report);
            reportHistoryContainer.appendChild(card);
        });
    }
    
    function createReportCard(report) {
        const card = document.createElement('div');
        card.className = 'report-card';
        card.dataset.id = report.id;
        
        // Format date
        const date = new Date(report.date);
        const formattedDate = date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        });
        
        // Add new badge if it's a new report
        if (report.isNew) {
            const newBadge = document.createElement('div');
            newBadge.className = 'new-report-badge';
            newBadge.textContent = 'NEW';
            card.appendChild(newBadge);
        }
        
        // Determine mode badge
        const modeBadgeHtml = report.mode === 'detailed' ? 
            `<div class="mode-badge-small mode-detailed"><i class="fas fa-book"></i> Detailed</div>` :
            `<div class="mode-badge-small mode-concise"><i class="fas fa-bolt"></i> Concise</div>`;
        
        // Create card content
        card.innerHTML = `
            <div class="report-card-header">
                <div class="report-date">
                    <i class="far fa-calendar-alt"></i>
                    ${formattedDate}
                </div>
                <h4 class="report-title">${report.title}</h4>
            </div>
            <div class="report-card-body">
                <div class="report-preview">${report.preview}</div>
                <div class="report-actions">
                    <div class="report-badge">
                        ${modeBadgeHtml}
                    </div>
                    <div class="view-report-btn">
                        <span>View ${report.mode === 'detailed' ? 'Report' : 'Response'}</span>
                        <i class="fas fa-arrow-right"></i>
                    </div>
                </div>
            </div>
        `;
        
        // Add click event to view the report
        card.addEventListener('click', function() {
            viewReport(report.id);
        });
        
        return card;
    }
    
    function viewReport(taskId) {
        if (!reportModal || !modalReportContent) return;
        
        // Show loading state in modal
        modalReportContent.innerHTML = `
            <div class="loading-container">
                <div class="spinner-large"></div>
                <p>Loading report...</p>
            </div>
        `;
        
        // Update download link
        if (modalDownloadBtn) {
            modalDownloadBtn.href = `/download/${taskId}`;
        }
        
        // Open modal
        reportModal.classList.add('open');
        document.body.style.overflow = 'hidden'; // Prevent body scrolling
        
        // Fetch report content with retry mechanism
        fetchReportWithRetry(taskId, 0);
    }
    
    // Add a retry mechanism for fetching reports to handle server restarts
    function fetchReportWithRetry(taskId, retryCount, maxRetries = 2) {
        fetch(`/report_content/${taskId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(reportData => {
                if (reportData.error) {
                    throw new Error(reportData.error);
                }
                
                // Set title and content
                if (modalReportTitle) {
                    modalReportTitle.textContent = reportData.title;
                    
                    // Add mode badge to title
                    const mode = reportData.mode || 'concise';
                    const modeBadgeClass = mode === 'detailed' ? 'mode-detailed' : 'mode-concise';
                    const modeIcon = mode === 'detailed' ? 'fas fa-book' : 'fas fa-bolt';
                    
                    // Add mode badge if not already present
                    if (!modalReportTitle.querySelector('.mode-badge-inline')) {
                        const modeBadge = document.createElement('span');
                        modeBadge.className = `mode-badge-inline ${modeBadgeClass}`;
                        modeBadge.innerHTML = `<i class="${modeIcon}"></i> ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
                        modalReportTitle.appendChild(modeBadge);
                    }
                }
                
                modalReportContent.innerHTML = reportData.html_content;
                
                // Load chat history in the modal if available
                if (modalChatMessages && reportData.chat_history && reportData.chat_history.length > 0) {
                    // Clear existing messages
                    modalChatMessages.innerHTML = '';
                    
                    // Add each message
                    reportData.chat_history.forEach(msg => {
                        if (msg.role !== 'system') {
                            const messageDiv = document.createElement('div');
                            messageDiv.className = `chat-message ${msg.role}`;
                            
                            const bubbleDiv = document.createElement('div');
                            bubbleDiv.className = `message-bubble ${msg.role}-bubble`;
                            
                            // Format content
                            let formattedContent = msg.content
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                                .replace(/`(.*?)`/g, '<code>$1</code>')
                                .replace(/\n/g, '<br>');
                            
                            bubbleDiv.innerHTML = formattedContent;
                            messageDiv.appendChild(bubbleDiv);
                            modalChatMessages.appendChild(messageDiv);
                        }
                    });
                    
                    // Show the chat container if we have messages
                    if (modalChatContainer && reportData.chat_history.length > 1) {
                        modalChatContainer.style.display = 'flex';
                    } else if (modalChatContainer) {
                        modalChatContainer.style.display = 'none';
                    }
                } else if (modalChatContainer) {
                    modalChatContainer.style.display = 'none';
                }
                
                // Apply syntax highlighting
                document.querySelectorAll('#modal-report-content pre code').forEach((block) => {
                    hljs.highlightBlock(block);
                });
                
                // Make tables responsive
                document.querySelectorAll('#modal-report-content table').forEach((table) => {
                    if (!table.parentNode.classList.contains('table-responsive')) {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'table-responsive';
                        table.parentNode.insertBefore(wrapper, table);
                        wrapper.appendChild(table);
                    }
                });
                
                // Mark as not new in history
                markReportAsViewed(taskId);
            })
            .catch(error => {
                console.error(`Error loading report (attempt ${retryCount + 1}):`, error);
                
                if (retryCount < maxRetries) {
                    // Show retry message
                    modalReportContent.innerHTML = `
                        <div class="loading-container">
                            <div class="spinner-large"></div>
                            <p>Retrying to load report... (${retryCount + 1}/${maxRetries + 1})</p>
                        </div>
                    `;
                    
                    // Retry after a delay
                    setTimeout(() => {
                        fetchReportWithRetry(taskId, retryCount + 1, maxRetries);
                    }, 1000);
                } else {
                    // Show error after all retries fail
                    modalReportContent.innerHTML = `
                        <div class="error-message">
                            <i class="fas fa-exclamation-triangle"></i>
                            Failed to load report: ${error.message}
                        </div>
                        <div style="margin-top: 20px; text-align: center;">
                            <p>The report may no longer be available on the server.</p>
                            <button id="remove-report-btn" class="action-button" style="margin-top: 15px;">
                                <i class="fas fa-trash"></i> Remove from history
                            </button>
                        </div>
                    `;
                    
                    // Add remove from history button functionality
                    const removeBtn = document.getElementById('remove-report-btn');
                    if (removeBtn) {
                        removeBtn.addEventListener('click', function() {
                            removeReportFromHistory(taskId);
                            closeReportModal();
                            showNotification('Report removed from history', 'info');
                        });
                    }
                }
            });
    }
    
    function removeReportFromHistory(taskId) {
        const reports = getReportHistory();
        const updatedReports = reports.filter(report => report.id !== taskId);
        saveReportHistory(updatedReports);
        displayReportHistory(updatedReports);
    }
    
    function closeReportModal() {
        if (!reportModal) return;
        
        reportModal.classList.remove('open');
        document.body.style.overflow = ''; // Restore body scrolling
        
        // Clear content after animation
        setTimeout(() => {
            if (!reportModal.classList.contains('open')) {
                if (modalReportContent) modalReportContent.innerHTML = '';
                if (modalReportTitle) {
                    // Remove mode badge if present
                    const modeBadge = modalReportTitle.querySelector('.mode-badge-inline');
                    if (modeBadge) {
                        modeBadge.remove();
                    }
                    modalReportTitle.textContent = 'Report';
                }
                if (modalChatMessages) modalChatMessages.innerHTML = '';
            }
        }, 300);
    }
    
    function markReportAsViewed(taskId) {
        const reports = getReportHistory();
        const reportIndex = reports.findIndex(r => r.id === taskId);
        
        if (reportIndex >= 0 && reports[reportIndex].isNew) {
            reports[reportIndex].isNew = false;
            saveReportHistory(reports);
            
            // Update display (in background)
            setTimeout(() => {
                displayReportHistory(reports);
            }, 100);
        }
    }
    
    function getReportHistory() {
        const historyJson = localStorage.getItem('plutoReportHistory');
        return historyJson ? JSON.parse(historyJson) : [];
    }
    
    function saveReportHistory(reports) {
        localStorage.setItem('plutoReportHistory', JSON.stringify(reports));
    }
    
    // Helper to extract a preview from the HTML content
    function extractPreviewFromHtml(htmlContent) {
        // Create a temporary div to parse the HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;
        
        // Try to get content from the executive summary
        let executiveSummary = null;
        const h2Elements = tempDiv.querySelectorAll('h2');
        for (let i = 0; i < h2Elements.length; i++) {
            if (h2Elements[i].textContent.includes('Executive Summary') || 
                h2Elements[i].textContent.includes('Abstract')) {
                executiveSummary = h2Elements[i];
                break;
            }
        }
        
        if (executiveSummary) {
            const nextParagraph = executiveSummary.nextElementSibling;
            if (nextParagraph && nextParagraph.tagName === 'P') {
                return nextParagraph.textContent.substring(0, 120) + '...';
            }
        }
        
        // Fallback to first paragraph
        const firstParagraph = tempDiv.querySelector('p');
        if (firstParagraph) {
            return firstParagraph.textContent.substring(0, 120) + '...';
        }
        
        return "Strategic analysis and recommendations...";
    }
    
    // Add CSS for spinner in modal
    const style = document.createElement('style');
    style.textContent = `
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 50px;
            color: var(--text-secondary);
        }
        
        .spinner-large {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .pulse-animation {
            animation: pulse 1s ease-in-out;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .scroll-hidden {
            opacity: 0 !important;
            visibility: hidden !important;
            transform: translateY(20px) !important;
            pointer-events: none !important;
        }
        
        .scrolling-up i {
            transform: rotate(180deg);
        }
        
        /* Mode selector styles */
        .mode-selector-container {
            display: flex;
            align-items: center;
            border-right: 1px solid var(--card-border);
            padding: 0 15px;
            margin-right: 10px;
        }
        
        .mode-selector {
            background-color: transparent;
            border: none;
            color: var(--text-primary);
            font-family: 'Poppins', sans-serif;
            font-size: 14px;
            cursor: pointer;
            padding: 8px 12px;
            border-radius: var(--border-radius-md);
            appearance: none;
            -webkit-appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23B0B0B0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 4px center;
            padding-right: 24px;
            transition: var(--transition-normal);
        }
        
        .mode-selector:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .mode-selector:focus {
            outline: none;
            box-shadow: 0 0 0 2px rgba(136, 136, 136, 0.3);
        }
        
        .mode-selector option {
            background-color: var(--surface-color);
            color: var(--text-primary);
        }
        
        /* Mode info tooltip */
        .mode-info-tooltip {
            position: relative;
            margin-left: 8px;
            color: var(--text-muted);
            cursor: pointer;
        }
        
        .mode-info-tooltip:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
            transform: translateY(0);
        }
        
        .tooltip-content {
            visibility: hidden;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(10px);
            width: 200px;
            background-color: var(--surface-color);
            border: 1px solid var(--card-border);
            border-radius: var(--border-radius-md);
            padding: 12px;
            color: var(--text-primary);
            box-shadow: var(--shadow-md);
            z-index: 100;
            opacity: 0;
            transition: opacity 0.3s, transform 0.3s;
            font-size: 12px;
        }
        
        .tooltip-option {
            margin-bottom: 8px;
        }
        
        .tooltip-option:last-child {
            margin-bottom: 0;
        }
        
        /* Mode badges */
        .mode-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            border-radius: var(--border-radius-md);
            font-size: 14px;
            font-weight: 500;
        }
        
        .mode-concise {
            background-color: rgba(138, 186, 174, 0.2);
            color: var(--success-color);
        }
        
        .mode-detailed {
            background-color: rgba(204, 204, 204, 0.2);
            color: var(--accent-color);
        }
        
        /* Status steps for concise mode */
        .status-steps.concise-mode .status-step:nth-child(3) {
            display: none;  /* Hide the last step for concise mode */
        }
        
        .status-steps.concise-mode .status-step:nth-child(2) .step-connector {
            display: none;  /* Hide connector to the third step */
        }
        
        /* Small mode badge for report cards */
        .mode-badge-small {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 10px;
        }
        
        /* Inline mode badge for modal title */
        .mode-badge-inline {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 14px;
            padding: 3px 10px;
            border-radius: 10px;
            margin-left: 10px;
            vertical-align: middle;
        }
    `;
    document.head.appendChild(style);
});

// Function to fill the query input with sample text
function fillQuery(text) {
    const queryInput = document.getElementById('query');
    if (!queryInput) return;
    
    queryInput.value = text;
    queryInput.focus();
    
    // Add a subtle pulse animation to the search button
    const searchBtn = document.getElementById('generate-btn');
    if (searchBtn) {
        searchBtn.classList.add('pulse-animation');
        
        // Remove the class after animation completes
        setTimeout(() => {
            searchBtn.classList.remove('pulse-animation');
        }, 1000);
    }
}
