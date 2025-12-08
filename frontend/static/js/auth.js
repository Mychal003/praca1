// ============================================================================
// AUTH MODULE
// ============================================================================

const API_URL = 'http://localhost:5000/api';

const Auth = {
    // Token management
    getToken() {
        return localStorage.getItem('auth_token');
    },

    setToken(token) {
        localStorage.setItem('auth_token', token);
    },

    removeToken() {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
    },

    getUser() {
        const data = localStorage.getItem('user_data');
        return data ? JSON.parse(data) : null;
    },

    setUser(user) {
        localStorage.setItem('user_data', JSON.stringify(user));
    },

    isLoggedIn() {
        return !!this.getToken();
    },

    // API headers with auth
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        const token = this.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    },

    // Login
    async login(username, password) {
        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Błąd logowania');
        }

        this.setToken(data.token);
        this.setUser(data.user);
        return data;
    },

    // Register
    async register(username, email, password) {
        const response = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Błąd rejestracji');
        }

        this.setToken(data.token);
        this.setUser(data.user);
        return data;
    },

    // Logout
    logout() {
        this.removeToken();
        window.location.reload();
    },

    // Verify token
    async verifyToken() {
        const token = this.getToken();
        if (!token) return false;

        try {
            const response = await fetch(`${API_URL}/auth/me`, {
                headers: this.getHeaders()
            });

            if (!response.ok) {
                this.removeToken();
                return false;
            }

            const data = await response.json();
            this.setUser(data.user);
            return true;
        } catch (error) {
            this.removeToken();
            return false;
        }
    }
};

// ============================================================================
// AUTH UI HANDLERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabs = document.querySelectorAll('.auth-tab');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            if (tab.dataset.tab === 'login') {
                loginForm.classList.add('active');
                registerForm.classList.remove('active');
            } else {
                registerForm.classList.add('active');
                loginForm.classList.remove('active');
            }
        });
    });

    // Login form submit
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const errorDiv = document.getElementById('loginError');
        errorDiv.textContent = '';

        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;

        try {
            const btn = loginForm.querySelector('button');
            btn.disabled = true;
            btn.textContent = 'Logowanie...';

            await Auth.login(username, password);
            showApp();
        } catch (error) {
            errorDiv.textContent = error.message;
        } finally {
            const btn = loginForm.querySelector('button');
            btn.disabled = false;
            btn.textContent = 'Zaloguj się';
        }
    });

    // Register form submit
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const errorDiv = document.getElementById('registerError');
        errorDiv.textContent = '';

        const username = document.getElementById('registerUsername').value.trim();
        const email = document.getElementById('registerEmail').value.trim();
        const password = document.getElementById('registerPassword').value;
        const passwordConfirm = document.getElementById('registerPasswordConfirm').value;

        // Validation
        if (password !== passwordConfirm) {
            errorDiv.textContent = 'Hasła nie są identyczne';
            return;
        }

        if (password.length < 6) {
            errorDiv.textContent = 'Hasło musi mieć minimum 6 znaków';
            return;
        }

        try {
            const btn = registerForm.querySelector('button');
            btn.disabled = true;
            btn.textContent = 'Rejestracja...';

            await Auth.register(username, email, password);
            showApp();
        } catch (error) {
            errorDiv.textContent = error.message;
        } finally {
            const btn = registerForm.querySelector('button');
            btn.disabled = false;
            btn.textContent = 'Zarejestruj się';
        }
    });

    // Logout button
    document.getElementById('logoutBtn').addEventListener('click', () => {
        if (confirm('Czy na pewno chcesz się wylogować?')) {
            Auth.logout();
        }
    });

    // Check if already logged in
    initApp();
});

// Initialize app based on auth state
async function initApp() {
    const authScreen = document.getElementById('authScreen');
    const appScreen = document.getElementById('appScreen');

    if (Auth.isLoggedIn()) {
        const isValid = await Auth.verifyToken();
        if (isValid) {
            showApp();
            return;
        }
    }

    // Show auth screen
    authScreen.style.display = 'flex';
    appScreen.style.display = 'none';
}

// Show main app
function showApp() {
    const authScreen = document.getElementById('authScreen');
    const appScreen = document.getElementById('appScreen');

    authScreen.style.display = 'none';
    appScreen.style.display = 'flex';

    // Update user info
    const user = Auth.getUser();
    if (user) {
        document.getElementById('userName').textContent = user.username;
    }

    // Load conversations - TYLKO RAZ
    if (typeof loadConversations === 'function') {
        loadConversations();
    }
}