# migrations.py

create_tables = """
CREATE TABLE IF NOT EXISTS messages (
    message_id BIGINT PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    text TEXT,
    answer_text TEXT NOT NULL Default ''
);

CREATE TABLE IF NOT EXISTS refs (
    message_id BIGINT NOT NULL,
    chat_id BIGINT NOT NULL,
    ref_link TEXT NOT NULL,
    ref_cite TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admins (
    chat_id BIGINT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    feedback INTEGER NOT NULL CHECK (feedback IN (0,1)),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS placeholders(
chat_id       BIGINT,
orig_msg_id   INTEGER,   
ref_number    INTEGER,   
placeholder_id INTEGER,
PRIMARY KEY (chat_id, orig_msg_id, ref_number)
);

"""

'''delete_redundant_messages = """
DELETE FROM messages 
WHERE chat_id = %s 
  AND message_id NOT IN (
    SELECT message_id 
    FROM messages 
    WHERE chat_id = %s 
    ORDER BY message_id DESC 
    LIMIT 3
);
"""'''
