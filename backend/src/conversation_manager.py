from models import db, Conversation, Message

class ConversationManager:
    """Zarządza konwersacjami użytkownika"""
    
    @staticmethod
    def create_conversation(user_id, document_name=None):
        """Tworzy nową konwersację"""
        conversation = Conversation(
            user_id=user_id,
            document_name=document_name,
            title=f"Rozmowa o {document_name}" if document_name else "Nowa rozmowa"
        )
        db.session.add(conversation)
        db.session.commit()
        return conversation
    
    @staticmethod
    def get_user_conversations(user_id, limit=20):
        """Pobiera konwersacje użytkownika"""
        return Conversation.query.filter_by(user_id=user_id)\
            .order_by(Conversation.updated_at.desc())\
            .limit(limit)\
            .all()
    
    @staticmethod
    def get_conversation(conversation_id, user_id):
        """Pobiera konkretną konwersację (z weryfikacją właściciela)"""
        return Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()
    
    @staticmethod
    def add_message(conversation_id, role, content, category=None, sources=None):
        """Dodaje wiadomość do konwersacji"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            category=category,
            sources=sources
        )
        db.session.add(message)
        
        # Aktualizuj tytuł konwersacji przy pierwszym pytaniu
        if role == 'user':
            conversation = Conversation.query.get(conversation_id)
            if conversation and len(conversation.messages) == 0:
                # Ustaw tytuł na podstawie pierwszego pytania
                conversation.title = content[:50] + ('...' if len(content) > 50 else '')
        
        db.session.commit()
        return message
    
    @staticmethod
    def delete_conversation(conversation_id, user_id):
        """Usuwa konwersację"""
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()
        if conversation:
            db.session.delete(conversation)
            db.session.commit()
            return True
        return False
    
    @staticmethod
    def update_title(conversation_id, user_id, new_title):
        """Aktualizuje tytuł konwersacji"""
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()
        if conversation:
            conversation.title = new_title
            db.session.commit()
            return True
        return False