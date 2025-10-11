import json
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from config import Config
from typing import List, Dict
from datetime import datetime

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


def get_curriculum(level, subject: str) -> List[Dict]:
    """Retrieve curriculum for a subject"""
    query = """
        SELECT id, topic, description, prerequisites, learning_objectives
        FROM eduai.curriculum
        WHERE subject = %s
        AND level = %s
    """
    return execute_query(query, (level, subject))

def save_progress(teacher_id: str, curriculum_id: int, topic: str, 
                  proficiency_level: str, knowledge_score: float, practical_score: float):
    """Save teacher's progress"""
    query = """
        INSERT INTO eduai.teacher_progress
        (teacher_id, curriculum_id, topic, proficiency_level, knowledge_score, practical_score, completed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    execute_query(query, (teacher_id, curriculum_id, topic, proficiency_level, 
                          knowledge_score, practical_score, datetime.now()), is_update=True)

def save_assessment(teacher_id: str, topic: str, assessment_type: str, result):
    """Save assessment results"""
    query = """
        INSERT INTO eduai.assessment_history
        (teacher_id, topic, assessment_type, score, strengths, weaknesses, recommendations)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    execute_query(query, (
        teacher_id,
        topic,
        assessment_type,
        result.score,
        json.dumps(result.strengths),
        json.dumps(result.weaknesses),
        json.dumps(result.recommendations)
    ), is_update=True)

def save_recommendations(teacher_id: str, recommendations: List):
    """Save learning recommendations"""
    query = """
        INSERT INTO eduai.learning_recommendations
        (teacher_id, topic, priority, actions, resources)
        VALUES (%s, %s, %s, %s, %s)
    """
    for rec in recommendations:
        execute_query(query, (
            teacher_id,
            rec.topic,
            rec.priority,
            json.dumps(rec.recommended_actions),
            json.dumps(rec.resources)
        ), is_update=True)

def get_teacher_progress(teacher_id: str) -> List[Dict]:
    """Get teacher's learning progress"""
    query = """
        SELECT topic, proficiency_level, knowledge_score, practical_score, completed_at
        FROM eduai.teacher_progress
        WHERE teacher_id = %s
        ORDER BY completed_at DESC
    """
    return execute_query(query, (teacher_id,))

def get_recommendations(teacher_id: str) -> List[Dict]:
    """Get pending recommendations"""
    query = """
        SELECT topic, priority, actions, resources, created_at
        FROM eduai.learning_recommendations
        WHERE teacher_id = %s AND status = 'pending'
        ORDER BY 
            CASE priority 
                WHEN 'High' THEN 1 
                WHEN 'Medium' THEN 2 
                ELSE 3 
            END
    """
    return execute_query(query, (teacher_id,))
