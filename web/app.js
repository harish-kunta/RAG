const form = document.querySelector('#chat-form');
const input = document.querySelector('#question');
const messages = document.querySelector('#messages');
const sendButton = document.querySelector('#send');
const statusBox = document.querySelector('.status');
const statusText = document.querySelector('#status-text');
const conversationList = document.querySelector('#conversation-list');
const newChatButton = document.querySelector('#new-chat');
let conversationId = null;

function setUiBusy(busy) {
  sendButton.disabled = busy;
  newChatButton.disabled = busy;
  for (const button of conversationList.querySelectorAll('button')) {
    button.disabled = busy;
  }
}

function showWelcome() {
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.id = 'welcome';

  const icon = document.createElement('div');
  icon.className = 'welcome-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = '✳';
  const heading = document.createElement('h2');
  heading.textContent = 'What would you like to know?';
  const description = document.createElement('p');
  description.textContent = 'Ask a question about the information you have indexed.';
  welcome.append(icon, heading, description);

  for (const suggestionText of [
    'What does the sample warranty cover?',
    'How long does domestic shipping take?',
  ]) {
    const suggestion = document.createElement('button');
    suggestion.className = 'suggestion';
    suggestion.type = 'button';
    suggestion.textContent = suggestionText;
    welcome.append(suggestion);
  }
  messages.replaceChildren(welcome);
}

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

function setSelectedConversation(id) {
  conversationId = id;
  for (const item of conversationList.querySelectorAll('.conversation-button')) {
    item.classList.toggle('selected', item.dataset.conversationId === id);
  }
}

function renderConversationList(conversations) {
  conversationList.replaceChildren();
  if (!conversations.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-history';
    empty.textContent = 'Your saved chats will appear here.';
    conversationList.append(empty);
    return;
  }

  for (const conversation of conversations) {
    const row = document.createElement('div');
    row.className = 'conversation-row';

    const openButton = document.createElement('button');
    openButton.className = 'conversation-button';
    openButton.type = 'button';
    openButton.dataset.conversationId = conversation.id;
    openButton.disabled = sendButton.disabled;
    openButton.classList.toggle('selected', conversation.id === conversationId);
    openButton.textContent = conversation.title;
    openButton.title = `${conversation.title} · ${conversation.message_count} messages`;

    const deleteButton = document.createElement('button');
    deleteButton.className = 'delete-conversation';
    deleteButton.type = 'button';
    deleteButton.dataset.deleteConversationId = conversation.id;
    deleteButton.disabled = sendButton.disabled;
    deleteButton.setAttribute('aria-label', `Delete ${conversation.title}`);
    deleteButton.title = 'Delete conversation';
    deleteButton.textContent = '×';

    row.append(openButton, deleteButton);
    conversationList.append(row);
  }
}

async function loadConversationList() {
  try {
    const response = await fetch('/api/conversations');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load saved chats.');
    renderConversationList(data.conversations);
  } catch (error) {
    conversationList.replaceChildren();
    const message = document.createElement('p');
    message.className = 'empty-history error-text';
    message.textContent = error.message;
    conversationList.append(message);
  }
}

async function openConversation(id) {
  if (sendButton.disabled) return;
  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not open this chat.');
    setSelectedConversation(data.id);
    messages.replaceChildren();
    if (!data.messages.length) showWelcome();
    for (const message of data.messages) {
      addMessage(message.role, message.content, message.sources);
    }
  } catch (error) {
    statusText.textContent = error.message;
    statusBox.classList.add('error');
  }
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    if (!health.vector_index_present) {
      statusText.textContent = 'Index needed — run python3 langchain_rag.py --index';
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
  input.value = '';
  input.style.height = 'auto';
  setUiBusy(true);
  const waiting = addMessage('assistant', 'Searching your sources…');

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The chat request failed.');
    waiting.remove();
    addMessage('assistant', data.answer, data.sources);
    setSelectedConversation(data.conversation_id);
    await loadConversationList();
  } catch (error) {
    waiting.querySelector('.bubble').textContent = error.message;
    statusText.textContent = 'Request failed';
    statusBox.classList.remove('ready');
    statusBox.classList.add('error');
  } finally {
    setUiBusy(false);
    input.focus();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

messages.addEventListener('click', (event) => {
  const suggestion = event.target.closest('.suggestion');
  if (!suggestion) return;
  input.value = suggestion.textContent;
  form.requestSubmit();
});

conversationList.addEventListener('click', async (event) => {
  if (sendButton.disabled) return;
  const deleteButton = event.target.closest('[data-delete-conversation-id]');
  if (deleteButton) {
    const id = deleteButton.dataset.deleteConversationId;
    if (!window.confirm('Delete this saved conversation?')) return;
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Could not delete this chat.');
      if (conversationId === id) {
        setSelectedConversation(null);
        showWelcome();
      }
      await loadConversationList();
    } catch (error) {
      statusText.textContent = error.message;
      statusBox.classList.add('error');
    }
    return;
  }

  const openButton = event.target.closest('[data-conversation-id]');
  if (openButton) await openConversation(openButton.dataset.conversationId);
});

newChatButton.addEventListener('click', () => {
  if (sendButton.disabled) return;
  setSelectedConversation(null);
  showWelcome();
  input.focus();
});

showWelcome();
checkHealth();
loadConversationList();
