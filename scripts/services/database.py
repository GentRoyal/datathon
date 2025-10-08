import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from config import Config

import logging
logger = logging.getLogger(__name__)

config = Config()
DATABASE_URL = config.DATABASE_URL

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        register_vector(conn)  
    except Exception as e:
        logger.exception("Failed to register pgvector on connection: %s", e)
    return conn

def execute_query(query, params = None, is_update = False):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or {})
            if is_update:
                conn.commit()
                return cur.rowcount
            else:
                return cur.fetchall()
    finally:
        conn.close()