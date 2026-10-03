"""Transactional local chat history, independent of indexes and experiments."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


class ChatStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    model TEXT NOT NULL, rag TEXT NOT NULL, draft TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL,
                    sources TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, id);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat()

    def create(self, model, rag):
        chat_id = uuid.uuid4().hex
        timestamp = self.timestamp()
        with self.connection() as db:
            db.execute('INSERT INTO chats VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (chat_id, 'New chat', timestamp, timestamp, model, rag, ''))
            db.execute("INSERT OR REPLACE INTO state VALUES ('active_chat', ?)", (chat_id,))
        return chat_id

    def list_chats(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute('SELECT * FROM chats ORDER BY updated_at DESC, id DESC')]

    def active(self):
        with self.connection() as db:
            row = db.execute("SELECT value FROM state WHERE key = 'active_chat'").fetchone()
            return row['value'] if row and db.execute('SELECT id FROM chats WHERE id = ?', (row['value'],)).fetchone() else None

    def set_active(self, chat_id):
        with self.connection() as db:
            if not db.execute('SELECT id FROM chats WHERE id = ?', (chat_id,)).fetchone():
                raise ValueError('Chat no longer exists.')
            db.execute("INSERT OR REPLACE INTO state VALUES ('active_chat', ?)", (chat_id,))

    def load(self, chat_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM chats WHERE id = ?', (chat_id,)).fetchone()
            if row is None:
                raise ValueError('Chat no longer exists.')
            chat = dict(row)
            chat['messages'] = [dict(message) | {'sources': json.loads(message['sources'])}
                                for message in db.execute('SELECT role, text, metadata, sources FROM messages WHERE chat_id = ? ORDER BY id', (chat_id,))]
            return chat

    def append(self, chat_id, role, text, metadata='', sources=None):
        with self.connection() as db:
            db.execute('INSERT INTO messages (chat_id, role, text, metadata, sources) VALUES (?, ?, ?, ?, ?)',
                       (chat_id, role, text, metadata, json.dumps(sources or [], ensure_ascii=False)))
            if role == 'You':
                title = ' '.join(text.split())[:70] or 'New chat'
                db.execute("UPDATE chats SET title = ? WHERE id = ? AND title = 'New chat'", (title, chat_id))
            db.execute('UPDATE chats SET updated_at = ? WHERE id = ?', (self.timestamp(), chat_id))

    def save_settings(self, chat_id, model, rag, draft):
        with self.connection() as db:
            db.execute('UPDATE chats SET model = ?, rag = ?, draft = ? WHERE id = ?', (model, rag, draft, chat_id))

    def delete(self, chat_id):
        with self.connection() as db:
            db.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
            db.execute("DELETE FROM state WHERE key = 'active_chat' AND value = ?", (chat_id,))
