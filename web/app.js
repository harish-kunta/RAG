const form = document.querySelector('#chat-form');
const input = document.querySelector('#question');
const messages = document.querySelector('#messages');
const sendButton = document.querySelector('#send');
const statusBox = document.querySelector('.status');
const statusText = document.querySelector('#status-text');
const history = [];

function addMessage(role, content, sources = []) {
  document.querySelector('#welcome')?.remove();
  const wrapper = document.createElement('article');
  wrapper.className = `message ${role}`;

  const label = document.createElement('p');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'RAG assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  wrapper.append(label, bubble);

  if (sources.length) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.textContent = `Sources used (${sources.length})`;
    details.append(summary);
    for (const source of sources) {
      const card = document.createElement('div');
      card.className = 'source-card';
      const title = document.createElement('p');
      title.className = 'source-title';
      let safeUrl = null;
      if (source.url) {
        try {
          const parsedUrl = new URL(source.url, window.location.origin);
          if (parsedUrl.protocol === 'https:' || parsedUrl.protocol === 'http:') {
            safeUrl = parsedUrl.href;
          }
        } catch {
          // A malformed source link is shown as plain text.
        }
      }
      if (safeUrl) {
        const link = document.createElement('a');
        link.href = safeUrl;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = source.label;
        title.append(link);
      } else {
        title.textContent = source.label;
      }
      const excerpt = document.createElement('p');
      excerpt.className = 'source-excerpt';
      excerpt.textContent = source.excerpt;
      card.append(title, excerpt);
      details.append(card);
    }
    wrapper.append(details);
  }

  messages.append(wrapper);
  messages.scrollTop = messages.scrollHeight;
  return wrapper;
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    if (!health.vector_index_present) {
      statusText.textContent = 'Index needed — run python3 rag.py --index';
      statusBox.classList.add('error');
      return;
    }
    statusText.textContent = 'Knowledge base ready';
    statusBox.classList.add('ready');
  } catch {
    statusText.textContent = 'Could not reach the chat server';
    statusBox.classList.add('error');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;

  addMessage('user', question);
  history.push({ role: 'user', content: question });
  input.value = '';
  input.style.height = 'auto';
  sendButton.disabled = true;
  const waiting = addMessage('assistant', 'Searching your sources…');

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: history.slice(0, -1).slice(-12) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The chat request failed.');
    waiting.remove();
    addMessage('assistant', data.answer, data.sources);
    history.push({ role: 'assistant', content: data.answer });
  } catch (error) {
    waiting.querySelector('.bubble').textContent = error.message;
    statusText.textContent = 'Request failed';
    statusBox.classList.remove('ready');
    statusBox.classList.add('error');
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

document.querySelectorAll('.suggestion').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.textContent;
    form.requestSubmit();
  });
});

document.querySelector('#new-chat').addEventListener('click', () => {
  history.length = 0;
  messages.replaceChildren();
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.id = 'welcome';
  welcome.innerHTML = '<div class="welcome-icon" aria-hidden="true">✳</div><h2>What would you like to know?</h2><p>Ask a question about the information you have indexed.</p>';
  messages.append(welcome);
  input.focus();
});

checkHealth();
