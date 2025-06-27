import psycopg2
from os import getenv

import logging
import migrations

def get_cursor() -> psycopg2.extensions.cursor:
    try:
        connection = psycopg2.connect(
        dbname   = getenv("POSTGRES_DB",     "db"),
        user     = getenv("POSTGRES_USER",   "postgres"),
        password = getenv("POSTGRES_PASSWORD","123"),
        host     = getenv("POSTGRES_HOST",   "localhost"),
        port     = getenv("POSTGRES_PORT",   5432),
    )
        connection.autocommit = True
        cursor = connection.cursor()
        return cursor
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

def migrate_postgres(SUPERADMIN) -> None:
    try:
        logging.info("Migrating postgres database...")
        cursor = get_cursor()
        cursor.execute(migrations.create_tables)
        cursor.execute("INSERT INTO admins (chat_id) VALUES (%s) ON CONFLICT DO NOTHING;", (str(SUPERADMIN),))
        cursor.close()
        logging.info("Postgres database migrated!")
    except Exception as e:
        logging.error(f"Migration error: {e}")

def add_message(message_id: int, chat_id: int, text: str, context, answer_text: str) -> None:
    try:
        cursor = get_cursor()
        cursor.execute(
            "INSERT INTO messages (message_id, chat_id, text, answer_text) VALUES (%s, %s, %s, %s)",
            (message_id, chat_id, text, str(answer_text))
        )
        cursor.execute("SELECT COUNT(*) FROM messages WHERE chat_id = %s", (chat_id,))
        '''if cursor.fetchone()[0] > 3:
            cursor.execute(migrations.delete_redundant_messages, (chat_id, chat_id))'''
        logging.info("Message added!")
        for doc in context:
            cursor.execute(
                "INSERT INTO refs (message_id, chat_id, ref_link, ref_cite) VALUES (%s, %s, %s, %s)",
                (message_id, chat_id, doc.metadata['link'], doc.page_content)
            )
            logging.info(f"ref added: chat_id={chat_id}, message_id={message_id}, ref_link={doc.metadata['link']},ref_cite={doc.page_content}")
        cursor.close()
    except Exception as e:
        logging.error(f"Message add error: {e}")

def save_placeholder(chat_id: int, orig_msg_id: int, ref_number: int, placeholder_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            """
            INSERT INTO placeholders(chat_id, orig_msg_id, ref_number, placeholder_id)
            VALUES (%s,      %s,          %s,         %s)
            ON CONFLICT (chat_id, orig_msg_id, ref_number)
            DO UPDATE SET placeholder_id = EXCLUDED.placeholder_id
            """,
            (chat_id, orig_msg_id, ref_number, placeholder_id)
        )
        cursor.close()
    except Exception as e:
        logging.error(f"Message add error: {e}")

def get_placeholder(chat_id: int, orig_msg_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            """
            SELECT placeholder_id
              FROM placeholders
             WHERE chat_id     = %s
               AND orig_msg_id = %s
            """,
            (chat_id, orig_msg_id)
        )
        row = cursor.fetchone()
        cursor.close()
        return row[0] if row else None
        
    except Exception as e:
        logging.error(f"Message add error: {e}")


def add_feedback(chat_id: int, message_id: int, feedback: int) -> None:
    try:
        cursor = get_cursor()
        cursor.execute(
            "INSERT INTO feedback (chat_id, message_id, feedback) VALUES (%s, %s, %s)",
            (chat_id, message_id, feedback)
        )
        cursor.close()
        logging.info(f"Feedback added: chat_id={chat_id}, message_id={message_id}, feedback={feedback}")
    except Exception as e:
        logging.error(f"Error adding feedback: {e}")

def get_ref(message_id: int, chat_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT ref_cite, ref_link FROM refs WHERE message_id = %s AND chat_id = %s",
            (message_id, chat_id)
        )
        row = cursor.fetchall()
        print(f"[DEBUG get_ref] row = {row}")
        return row
    except Exception as e:
        logging.error(f"Error fetching reference: {e}")
        return None

def get_all_admins():
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT * FROM admins"
        )
        row = cursor.fetchall()
        logging.info(f"[DEBUG get_admin] row = {row}")
        return row
    except Exception as e:
        logging.error(f"Error fetching reference: {e}")
        return None
    
def get_all_ref(message_id: int, chat_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT DISTINCT ref_link FROM refs WHERE message_id = %s AND chat_id = %s",
            (message_id, chat_id)
        )
        row = [item[0] for item in cursor.fetchall()]
        logging.info(f"[DEBUG get_all_ref] row = {row}")
        return row
    except Exception as e:
        logging.error(f"Error fetching reference: {e}")
        return None

def get_ref_count(message_id: int, chat_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT count(*) FROM refs WHERE message_id = %s AND chat_id = %s LIMIT 1",
            (message_id, chat_id)
        )
        row = cursor.fetchone()
        logging.info(f"[DEBUG count_ref] row = {row}")
        return row
    except Exception as e:
        logging.error(f"Error fetching reference: {e}")
        return None
def get_answer(message_id: int, chat_id: int):
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT answer_text FROM messages WHERE message_id = %s AND chat_id = %s LIMIT 1",
            (message_id, chat_id)
        )
        row = cursor.fetchone()
        logging.info(f"Message get: text={row}")
        return row
    except Exception as e:
        logging.error(f"Error fetching reference: {e}")
        return None

def is_admin(chat_id: int) -> bool:
    try:
        cursor = get_cursor()
        cursor.execute("SELECT COUNT(*) FROM admins WHERE chat_id = %s", (chat_id,))
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    except Exception as e:
        logging.error(f"Error checking admin status: {e}")
        return False

def add_admin(chat_id: int) -> None:
    try:
        cursor = get_cursor()
        cursor.execute(
            "INSERT INTO admins (chat_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (chat_id,)
        )
        cursor.close()
        logging.info(f"Admin added: chat_id={chat_id}")
    except Exception as e:
        logging.error(f"Error adding admin: {e}")

def get_previous_message(chat_id: int) -> str:
    try:
        cursor = get_cursor()
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = %s ORDER BY message_id DESC LIMIT 1",
            (chat_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else ''
    except Exception as e:
        logging.error(f"Error fetching previous message: {e}")
        return ''

def delete_admin(chat_id: int) -> None:
    try:
        cursor = get_cursor()
        cursor.execute("DELETE FROM admins WHERE chat_id = %s", (chat_id,))
        cursor.close()
        logging.info(f"Admin deleted: chat_id={chat_id}")
    except Exception as e:
        logging.error(f"Error deleting admin: {e}")
