"""
Prompt templates for French NLI tasks.
"""

from typing import List, Dict


class NLIPromptTemplate:
    """Templates for NLI prompting in French."""
    
    # Zero-shot template
    ZERO_SHOT_TEMPLATE = """Tâche : Inférence en Langage Naturel (NLI)

Étant donné une prémisse et une hypothèse, détermine la relation logique entre elles.

Réponse possible :
- entailment : l'hypothèse découle logiquement de la prémisse
- contradiction : l'hypothèse contredit la prémisse
- neutral : aucune relation logique claire

Prémisse : {premise}
Hypothèse : {hypothesis}

Réponse (entailment, contradiction ou neutral) :"""

    # Few-shot template
    FEW_SHOT_TEMPLATE = """Tâche : Inférence en Langage Naturel (NLI)

Étant donné une prémisse et une hypothèse, détermine la relation logique entre elles.

Réponse possible :
- entailment : l'hypothèse découle logiquement de la prémisse
- contradiction : l'hypothèse contredit la prémisse
- neutral : aucune relation logique claire

Exemples :

{examples}

Maintenant, résous cette instance :

Prémisse : {premise}
Hypothèse : {hypothesis}

Réponse (entailment, contradiction ou neutral) :"""

    # Example template for few-shot
    EXAMPLE_TEMPLATE = """Prémisse : {premise}
Hypothèse : {hypothesis}
Réponse : {label}"""

    @classmethod
    def format_zero_shot(cls, premise: str, hypothesis: str) -> str:
        """
        Format a zero-shot prompt.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            
        Returns:
            Formatted prompt string
        """
        return cls.ZERO_SHOT_TEMPLATE.format(
            premise=premise,
            hypothesis=hypothesis
        )
    
    @classmethod
    def format_few_shot(
        cls,
        premise: str,
        hypothesis: str,
        examples: List[Dict[str, str]]
    ) -> str:
        """
        Format a few-shot prompt.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            examples: List of example dicts with 'premise', 'hypothesis', 'label'
            
        Returns:
            Formatted prompt string
        """
        examples_text = "\n\n".join([
            cls.EXAMPLE_TEMPLATE.format(**ex)
            for ex in examples
        ])
        
        return cls.FEW_SHOT_TEMPLATE.format(
            examples=examples_text,
            premise=premise,
            hypothesis=hypothesis
        )
    
    @classmethod
    def format_chat_messages(
        cls,
        premise: str,
        hypothesis: str,
        examples: List[Dict[str, str]] = None,
        model_type: str = 'gpt'
    ) -> List[Dict[str, str]]:
        """
        Format prompts as chat messages for API models.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            examples: Optional few-shot examples
            model_type: Type of model ('gpt', 'gemini', 'mistral')
            
        Returns:
            List of message dicts for chat API
        """
        system_message = {
            'role': 'system',
            'content': (
                "Tu es un expert en inférence en langage naturel (NLI). "
                "Ta tâche est de déterminer la relation logique entre une "
                "prémisse et une hypothèse. Réponds uniquement par : "
                "entailment, contradiction, ou neutral."
            )
        }
        
        messages = [system_message]
        
        # Add few-shot examples as conversation history
        if examples:
            for ex in examples:
                user_msg = {
                    'role': 'user',
                    'content': f"Prémisse : {ex['premise']}\nHypothèse : {ex['hypothesis']}"
                }
                assistant_msg = {
                    'role': 'assistant',
                    'content': ex['label']
                }
                messages.extend([user_msg, assistant_msg])
        
        # Add the actual query
        query_msg = {
            'role': 'user',
            'content': f"Prémisse : {premise}\nHypothèse : {hypothesis}"
        }
        messages.append(query_msg)
        
        return messages


class PromptBuilder:
    """Builder for constructing NLI prompts with various configurations."""
    
    def __init__(self, template_class=NLIPromptTemplate):
        self.template = template_class
        
    def build_prompt(
        self,
        premise: str,
        hypothesis: str,
        examples: List[Dict[str, str]] = None,
        format_type: str = 'text'
    ) -> str:
        """
        Build a prompt for NLI.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            examples: Optional few-shot examples
            format_type: 'text' or 'chat'
            
        Returns:
            Formatted prompt (string or list of messages)
        """
        if format_type == 'text':
            if examples:
                return self.template.format_few_shot(premise, hypothesis, examples)
            else:
                return self.template.format_zero_shot(premise, hypothesis)
                
        elif format_type == 'chat':
            return self.template.format_chat_messages(premise, hypothesis, examples)
            
        else:
            raise ValueError(f"Unknown format_type: {format_type}")


if __name__ == '__main__':
    # Example usage
    prompt_builder = PromptBuilder()
    
    # Zero-shot example
    print("=" * 50)
    print("ZERO-SHOT PROMPT")
    print("=" * 50)
    zero_shot = prompt_builder.build_prompt(
        premise="Un homme joue de la guitare.",
        hypothesis="Un musicien fait de la musique."
    )
    print(zero_shot)
    
    # Few-shot example
    print("\n" + "=" * 50)
    print("FEW-SHOT PROMPT")
    print("=" * 50)
    examples = [
        {
            'premise': "Un chien court dans le parc.",
            'hypothesis': "Un animal est à l'extérieur.",
            'label': "entailment"
        },
        {
            'premise': "Une femme lit un livre.",
            'hypothesis': "Une personne regarde la télévision.",
            'label': "contradiction"
        },
        {
            'premise': "Il pleut aujourd'hui.",
            'hypothesis': "Le ciel est nuageux.",
            'label': "neutral"
        }
    ]
    
    few_shot = prompt_builder.build_prompt(
        premise="Un homme joue de la guitare.",
        hypothesis="Un musicien fait de la musique.",
        examples=examples[:2]  # 2-shot
    )
    print(few_shot)
    
    # Chat format
    print("\n" + "=" * 50)
    print("CHAT FORMAT")
    print("=" * 50)
    chat_messages = prompt_builder.build_prompt(
        premise="Un homme joue de la guitare.",
        hypothesis="Un musicien fait de la musique.",
        examples=examples[:2],
        format_type='chat'
    )
    
    for msg in chat_messages:
        print(f"{msg['role'].upper()}: {msg['content']}\n")
