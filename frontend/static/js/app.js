// ============================================================================
// APP MODULE - Conversations & Chat
// ============================================================================

// Current state
let currentConversationId = null;
let currentDocumentName = null;

// ============================================================================
// API CALLS
// ============================================================================

async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        headers: Auth.getHeaders()
    };
await loadConversations()
    const response = await fetch(`${API_URL}${endpoint}`, {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    });

    if (response.status === 401) {
        Auth.logout();
        throw new Error('Sesja wygasła');
    }

    return response;
}

// ============================================================================
// CONVERSATIONS
// ============================================================================

let isLoadingConversations = false; // NOWE: blokada przed wielokrotnym wywołaniem

async function loadConversations() {
    // Zapobiegaj wielokrotnym równoczesnym wywołaniom
    if (isLoadingConversations) {
        return;
    }
    
    isLoadingConversations = true;
    
    try {
        const response = await apiCall('/conversations');
        const data = await response.json();

        if (response.ok) {
            renderConversationsList(data.conversations);
        }
    } catch (error) {
        console.error('Błąd ładowania konwersacji:', error);
        showToast('Błąd ładowania konwersacji', 'error');
    } finally {
        isLoadingConversations = false;
    }
}

function renderConversationsList(conversations) {
    const listEl = document.getElementById('conversationsList');

    if (conversations.length === 0) {
        listEl.innerHTML = `
            <div class="empty-conversations">
                <p>Brak rozmów</p>
                <small>Kliknij + aby rozpocząć nową rozmowę</small>
            </div>
        `;
        return;
    }

    listEl.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
             data-id="${conv.id}">
            <div class="conversation-info">
                <span class="conversation-title">${escapeHtml(conv.title)}</span>
                <span class="conversation-meta">
                    ${conv.document_name ? `📄 ${conv.document_name}` : 'Brak dokumentu'}
                    · ${conv.message_count} wiad.
                </span>
            </div>
            <button class="btn-delete-conv" data-id="${conv.id}" title="Usuń">×</button>
        </div>
    `).join('');

    // Click handlers
    listEl.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('btn-delete-conv')) {
                openConversation(parseInt(item.dataset.id));
            }
        });
    });

    // Delete handlers
    listEl.querySelectorAll('.btn-delete-conv').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteConversation(parseInt(btn.dataset.id));
        });
    });
}

async function createNewConversation() {
    try {
        const response = await apiCall('/conversations', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            currentConversationId = data.conversation.id;
            currentDocumentName = null;
            
            await loadConversations();
            resetChatUI();
            showToast('Utworzono nową rozmowę', 'success');
        }
    } catch (error) {
        console.error('Błąd tworzenia konwersacji:', error);
        showToast('Błąd tworzenia rozmowy', 'error');
    }
}

async function openConversation(conversationId) {
    try {
        const response = await apiCall(`/conversations/${conversationId}`);
        const data = await response.json();

        if (response.ok) {
            currentConversationId = conversationId;
            currentDocumentName = data.conversation.document_name;

            // Update UI
            updateActiveConversation();
            renderMessages(data.conversation.messages);

            // Show/hide upload or chat
            if (currentDocumentName) {
                document.getElementById('uploadSection').style.display = 'none';
                document.getElementById('chatSection').style.display = 'flex';
                document.getElementById('currentDocName').textContent = `Dokument: ${currentDocumentName}`;
            } else {
                document.getElementById('uploadSection').style.display = 'flex';
                document.getElementById('chatSection').style.display = 'none';
                document.getElementById('currentDocName').textContent = 'Wgraj dokument, aby rozpocząć';
            }
        }
    } catch (error) {
        console.error('Błąd otwierania konwersacji:', error);
        showToast('Błąd otwierania rozmowy', 'error');
    }
}

async function deleteConversation(conversationId) {
    if (!confirm('Czy na pewno chcesz usunąć tę rozmowę?')) return;

    try {
        const response = await apiCall(`/conversations/${conversationId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            if (currentConversationId === conversationId) {
                currentConversationId = null;
                currentDocumentName = null;
                resetChatUI();
            }
            await loadConversations();
            showToast('Rozmowa usunięta', 'success');
        }
    } catch (error) {
        console.error('Błąd usuwania konwersacji:', error);
        showToast('Błąd usuwania rozmowy', 'error');
    }
}

function updateActiveConversation() {
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.id) === currentConversationId);
    });
}

function resetChatUI() {
    document.getElementById('chatMessages').innerHTML = '';
    document.getElementById('uploadSection').style.display = 'flex';
    document.getElementById('chatSection').style.display = 'none';
    document.getElementById('currentDocName').textContent = 'Wgraj dokument, aby rozpocząć';
    document.getElementById('pdfFile').value = '';
    document.getElementById('uploadStatus').innerHTML = '';
}

// ============================================================================
// FILE UPLOAD
// ============================================================================

async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];

    if (!file) {
        showToast('Wybierz plik PDF', 'warning');
        return;
    }

    // NOWE: Jeśli nie ma aktywnej konwersacji, utwórz ją automatycznie
    if (!currentConversationId) {
        try {
            const response = await apiCall('/conversations', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (response.ok) {
                currentConversationId = data.conversation.id;
                await loadConversations();
            } else {
                showToast('Błąd tworzenia rozmowy', 'error');
                return;
            }
        } catch (error) {
            showToast('Błąd tworzenia rozmowy', 'error');
            return;
        }
    }

    const formData = new FormData();
    formData.append('file', file);

    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = '<p class="loading-text"><span class="loading"></span> Przetwarzam dokument...</p>';

    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/conversations/${currentConversationId}/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${Auth.getToken()}`
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentDocumentName = data.filename;
            statusDiv.innerHTML = `<p class="success">✅ Dokument gotowy! (${data.processing_time}s)</p>`;
            
            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('chatSection').style.display = 'flex';
            document.getElementById('currentDocName').textContent = `Dokument: ${data.filename}`;
            
            await loadConversations();
            showToast('Dokument przetworzony pomyślnie', 'success');
        } else {
            statusDiv.innerHTML = `<p class="error">❌ ${data.error}</p>`;
            showToast(data.error, 'error');
        }
    } catch (error) {
        statusDiv.innerHTML = `<p class="error">❌ Błąd: ${error.message}</p>`;
        showToast('Błąd połączenia z serwerem', 'error');
    } finally {
        uploadBtn.disabled = false;
    }
}

// ============================================================================
// CHAT
// ============================================================================

function renderMessages(messages) {
    const messagesDiv = document.getElementById('chatMessages');

    if (!messages || messages.length === 0) {
        messagesDiv.innerHTML = '';
        return;
    }

    messagesDiv.innerHTML = messages.map(msg => createMessageHTML(msg)).join('');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function createMessageHTML(msg) {
    let html = `<div class="message ${msg.role}">`;
    html += `<p>${escapeHtml(msg.content)}</p>`;

    if (msg.category) {
        html += `<small>Kategoria: ${msg.category}</small>`;
    }

    if (msg.sources && msg.sources.length > 0) {
        html += '<details><summary>📖 Źródła</summary><ul>';
        msg.sources.forEach(src => {
            const text = src.text || src;
            html += `<li>${escapeHtml(text.substring(0, 150))}...</li>`;
        });
        html += '</ul></details>';
    }

    html += '</div>';
    return html;
}

async function askQuestion() {
    if (!currentConversationId) {
        showToast('Wybierz lub utwórz rozmowę', 'warning');
        return;
    }

    const input = document.getElementById('questionInput');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    addMessageToUI(question, 'user');
    input.value = '';

    // Loading state
    const loadingId = addMessageToUI('⏳ Szukam odpowiedzi...', 'assistant');

    try {
        const response = await apiCall(`/conversations/${currentConversationId}/query`, {
            method: 'POST',
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        // Remove loading
        document.getElementById(loadingId)?.remove();

        if (response.ok) {
            addMessageToUI(data.answer, 'assistant', data.category, data.sources);
            await loadConversations(); // Refresh list (update message count)
        } else {
            addMessageToUI(`Błąd: ${data.error}`, 'error');
        }
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        addMessageToUI(`Błąd: ${error.message}`, 'error');
    }
}

function addMessageToUI(text, role, category = null, sources = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageId = `msg-${Date.now()}`;

    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message ${role}`;

    let html = `<p>${escapeHtml(text)}</p>`;

    if (category) {
        html += `<small>Kategoria: ${category}</small>`;
    }

    if (sources && sources.length > 0) {
        html += '<details><summary>📖 Źródła</summary><ul>';
        sources.forEach(src => {
            const srcText = src.text || src;
            html += `<li>${escapeHtml(srcText.substring(0, 150))}...</li>`;
        });
        html += '</ul></details>';
    }

    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return messageId;
}

// ============================================================================
// UI UTILITIES
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // New chat button
    document.getElementById('newChatBtn').addEventListener('click', createNewConversation);

    // Upload button - POPRAWIONE
    document.getElementById('uploadBtn').addEventListener('click', () => {
        document.getElementById('pdfFile').click(); // Otwórz okno wyboru pliku
    });

    // File input change - POPRAWIONE: automatyczny upload po wybraniu pliku
    document.getElementById('pdfFile').addEventListener('change', async (e) => {
        if (e.target.files[0]) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = `<p class="info">📄 Wybrano: ${e.target.files[0].name}</p>`;
            
            // Automatycznie wgraj plik
            await uploadPDF();
        }
    });

    // Send button
    document.getElementById('sendBtn').addEventListener('click', askQuestion);

    // Enter to send
    document.getElementById('questionInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askQuestion();
    });

    // Toggle sidebar (mobile)
    document.getElementById('toggleSidebar').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });

    // Kliknięcie w upload-box też otwiera wybór pliku
    const uploadBox = document.querySelector('.upload-box');
    if (uploadBox) {
        uploadBox.addEventListener('click', (e) => {
            // Nie otwieraj jeśli kliknięto w przycisk (przycisk ma swój handler)
            if (e.target.tagName !== 'BUTTON') {
                document.getElementById('pdfFile').click();
            }
        });

        // Drag and drop
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('drag-over');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('drag-over');
        });

        uploadBox.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadBox.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.pdf')) {
                document.getElementById('pdfFile').files = files;
                document.getElementById('uploadStatus').innerHTML = 
                    `<p class="info">📄 Wybrano: ${files[0].name}</p>`;
                await uploadPDF();
            } else {
                showToast('Wybierz plik PDF', 'warning');
            }
        });
    }
});