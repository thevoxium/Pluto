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
    
    // Timer variables
    let startTime;
    let timerInterval;
    let currentTaskId = null;
    
    // Load report history on page load
    loadReportHistory();
    
    if (reportForm) {
        reportForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value.trim();
            if (!query) {
                showNotification('Please enter a research query', 'error');
                return;
            }
            
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
            progressIndicator.textContent = '1/7';
            currentQuery.textContent = query;
            
            // Reset research UI
            resetResearchUI();
            
            // Start timer
            startTimer();
            
            // Set initial status
            setStatus('thinking');
            
            // Submit form data
            const formData = new FormData(reportForm);
            
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
    
    function pollTaskStatus(taskId) {
        // Store for adding to history later
        let reportTitle = "";
        
        const pollInterval = setInterval(() => {
            fetch(`/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        stopTimer();
                        
                        // Complete all status steps
                        setStatus('completed');
                        
                        // When completed, fetch the report content and show it in the same page
                        fetch(`/report_content/${taskId}`)
                            .then(response => response.json())
                            .then(reportData => {
                                if (reportData.error) {
                                    throw new Error(reportData.error);
                                }
                                
                                // Save report title for history
                                reportTitle = reportData.title;
                                
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
                                addReportToHistory(taskId, reportTitle, previewText);
                                
                                // Scroll to report section with smooth animation
                                reportSection.scrollIntoView({ behavior: 'smooth' });
                                
                                // Show completion notification
                                showNotification('Report generated successfully!', 'success');
                            })
                            .catch(error => {
                                console.error('Error loading report:', error);
                                showError('Failed to load report: ' + error.message);
                            });
                    } else if (data.status === 'failed') {
                        clearInterval(pollInterval);
                        stopTimer();
                        showError('Report generation failed: ' + (data.error || 'Unknown error'));
                    } else if (data.status === 'running') {
                        // Update UI with progress info
                        updateResearchUI(data);
                    }
                })
                .catch(error => {
                    console.error('Error polling status:', error);
                    // Don't stop polling on network errors
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
                    if (data.progress.current_step <= 2) {
                        setStatus('thinking');
                    } else if (data.progress.current_step <= 5) {
                        setStatus('research');
                    } else {
                        setStatus('analysis');
                    }
                }
            } else {
                percent = data.progress;
            }
            
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
    function addReportToHistory(taskId, title, previewText) {
        if (!reportHistoryContainer) return;
        
        const reports = getReportHistory();
        
        // Create new report object
        const newReport = {
            id: taskId,
            title: title,
            preview: previewText || "Strategic analysis and recommendations for " + title,
            date: new Date().toISOString(),
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
                        <i class="fas fa-file-lines"></i>
                        <span>Report</span>
                    </div>
                    <div class="view-report-btn">
                        <span>View Report</span>
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
        
        // Fetch report content
        fetch(`/report_content/${taskId}`)
            .then(response => response.json())
            .then(reportData => {
                if (reportData.error) {
                    throw new Error(reportData.error);
                }
                
                // Set title and content
                if (modalReportTitle) {
                    modalReportTitle.textContent = reportData.title;
                }
                modalReportContent.innerHTML = reportData.html_content;
                
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
                modalReportContent.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle"></i>
                        Failed to load report: ${error.message}
                    </div>
                `;
            });
    }
    
    function closeReportModal() {
        if (!reportModal) return;
        
        reportModal.classList.remove('open');
        document.body.style.overflow = ''; // Restore body scrolling
        
        // Clear content after animation
        setTimeout(() => {
            if (!reportModal.classList.contains('open')) {
                if (modalReportContent) modalReportContent.innerHTML = '';
                if (modalReportTitle) modalReportTitle.textContent = 'Report';
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
            if (h2Elements[i].textContent.includes('Executive Summary')) {
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
