from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from .query_engine import QueryEngine, QueryResult

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        return self.role == other.role and self.content == other.content

    def __hash__(self):
        return hash((self.role, self.content))

@dataclass
class ConversationContext:
    messages: List[Message]
    last_query_results: Optional[List[QueryResult]] = None
    system_message_added: bool = False

class ConversationManager:
    def __init__(self, query_engine: QueryEngine, api_key: str):
        """Initialize conversation manager with query engine and OpenAI credentials."""
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")
        self.query_engine = query_engine
        self.client = OpenAI(api_key=api_key)
        
        # Updated comprehensive system prompt based on Asa's guidelines
        self.system_prompt: str = """You are Asa, an experienced principal broker providing strategic guidance to real estate agents. Your role is to offer expert advice on property transactions while maintaining a structured, analytical approach focused on educational value.

Key Behavioral Guidelines:

1. First Response Rule:
- Keep initial responses under 75 words
- Always end with a relevant question
- Never say "How can I assist you today?" - you are a collaborator, not an assistant

2. Communication Style:
- Maintain a professional, measured tone
- Never use exclamation marks
- Avoid absolutes like "Certainly," "Absolutely," or "Great"
- Use conditional language reflecting the complexity of real estate
- Structure responses with clear sections and periodic summaries

3. Citation and Information Requirements:
- ALWAYS cite sources using [Source] notation immediately after each statement
- Include citations inline, not at the end of the response
- Base all factual claims on provided documents
- When multiple documents support a point, cite all relevant sources

4. Strategic Focus:
- Always reference and cite information from provided documents
- Ask clarifying questions before offering comprehensive advice
- Emphasize compliance with laws and regulations
- Recommend consulting specialists when appropriate
- Provide data-driven insights and market analysis

5. Educational Approach:
- Explain underlying principles beyond immediate answers
- Detail applicable best practices
- Include relevant citations and their context
- Structure complex legal or regulatory information sequentially

6. Engagement Style:
- Act as a knowledgeable collaborator (think Obi-Wan Kenobi)
- Focus on strategic guidance and expertise
- Maintain a proactive, analytical dialogue
- Adapt responses based on user's level of understanding

Citation Format Example:
"The property market in Winslow has seen significant growth [Winslow-Report], with average prices increasing by 15% [Market-Analysis]. Recent developments have added 200 new units to the area [Development-Update]."

Remember: Your goal is to provide strategic, educational value while maintaining professional standards and encouraging thorough analysis of real estate scenarios."""
    
    def _format_context(self, query_results: List[QueryResult]) -> str:
        """Format retrieved documents into context string with citations."""
        context_parts: List[str] = []
        for result in query_results:
            source = str(result.metadata['source']).replace('.txt', '')  # Remove .txt extension
            context_parts.append(f"[{source}] {result.text}")
        return "\n\n".join(context_parts)
    
    def _create_message(self, role: Literal["system", "user", "assistant"], content: str) -> ChatCompletionMessageParam:
        """Create a properly-typed message for the OpenAI API."""
        if role == "system":
            return ChatCompletionSystemMessageParam(role=role, content=content)
        elif role == "user":
            return ChatCompletionUserMessageParam(role=role, content=content)
        else:
            return ChatCompletionAssistantMessageParam(role=role, content=content)
    
    def get_response(self, query: str, context: ConversationContext) -> str:
        """Generate a response using GPT-4 based on query and conversation context."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Add system message if not already added
        if not context.system_message_added:
            context.messages.append(Message(role="system", content=self.system_prompt))
            context.system_message_added = True
        
        # Add user message
        context.messages.append(Message(role="user", content=query))
        
        # Retrieve relevant documents
        query_results = self.query_engine.query(query)
        context.last_query_results = query_results
        
        # Format document context
        doc_context = self._format_context(query_results)
        
        # Prepare messages for GPT-4
        messages: List[ChatCompletionMessageParam] = []
        
        # Add system message
        messages.append(self._create_message("system", self.system_prompt))
        
        # Add conversation history (excluding system message)
        for msg in context.messages:
            if msg.role != "system":
                messages.append(self._create_message(msg.role, msg.content))
        
        # Add document context to the last message
        messages[-1] = self._create_message(
            "user",
            f"""{query}

Relevant Documents:
{doc_context}

Remember: 
1. If this is the user's first query, keep your response under 75 words and end with a question
2. ALWAYS include inline citations using [Source] format after each statement or claim
3. Each factual statement must have a citation
4. Do not group citations at the end - place them immediately after relevant information
5. Maintain professional tone without exclamation marks or absolutes"""
        )
        
        # Get completion from GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        if not response.choices:
            raise ValueError("No response received from OpenAI")
            
        generated_response = str(response.choices[0].message.content)
        if not generated_response:
            raise ValueError("Empty response received from OpenAI")
        
        # Add assistant response to context
        context.messages.append(Message(role="assistant", content=generated_response))
        
        return generated_response

class SessionManager:
    """Manage conversation sessions in Streamlit."""
    
    @staticmethod
    def initialize_session(st) -> None:
        """Initialize session state variables."""
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = ConversationContext(messages=[], system_message_added=False)
    
    @staticmethod
    def get_conversation_context(st) -> ConversationContext:
        """Retrieve current conversation context."""
        return st.session_state.conversation_context
