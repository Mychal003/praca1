const API_URL = 'http://localhost:5000/api';

async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Wybierz plik PDF');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = '<p>⏳ Przetwarzam dokument...</p>';
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            statusDiv.innerHTML = '<p style="color: green;">✅ Dokument gotowy!</p>';
            document.getElementById('chatSection').style.display = 'block';
            document.getElementById('uploadSection').style.display = 'none';
        } else {
            statusDiv.innerHTML = `<p style="color: red;">❌ Błąd: ${data.error}</p>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<p style="color: red;">❌ Błąd: ${error}</p>`;
    }
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    // Wyświetl pytanie użytkownika
    addMessage(question, 'user');
    input.value = '';
    
    // Pokaż loading
    const loadingId = addMessage('⏳ Szukam odpowiedzi...', 'assistant');
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question})
        });
        
        const data = await response.json();
        
        // Usuń loading i pokaż odpowiedź
        document.getElementById(loadingId).remove();
        
        if (response.ok) {
            addMessage(data.answer, 'assistant', data.category, data.sources);
        } else {
            addMessage(`Błąd: ${data.error}`, 'error');
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage(`Błąd: ${error}`, 'error');
    }
}

function addMessage(text, role, category = null, sources = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageId = `msg-${Date.now()}`;
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message ${role}`;
    
    let html = `<p>${text}</p>`;
    
    if (category) {
        html += `<small>Kategoria: ${category}</small>`;
    }
    
    if (sources && sources.length > 0) {
        html += '<details><summary>📖 Źródła</summary><ul>';
        sources.forEach(src => {
            html += `<li>${src.text.substring(0, 150)}...</li>`;
        });
        html += '</ul></details>';
    }
    
    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return messageId;
}

// Enter = wyślij pytanie
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('questionInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askQuestion();
    });
});