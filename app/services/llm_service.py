import openai
import json
from typing import List, Dict
from app.config import config

class LLMService:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set")
        openai.api_key = config.OPENAI_API_KEY
    
    async def match_question_to_classes(self, question: str, 
                                       classes: List[Dict]) -> List[Dict[str, float]]:
        """Use LLM to match question to available classes when vector search fails"""
        
        if not classes:
            return []
        
        # Prepare class descriptions for the LLM
        class_descriptions = []
        for cls in classes:
            class_descriptions.append(
                f"Class ID: {cls['class_id']}\n"
                f"Class Name: {cls['class_name']}\n"
            )
        
        prompt = f"""
Given the following question and available classes, determine which classes are most relevant.
Return a JSON object with class_id as keys and probability scores as values (0.0 to 1.0).
The probabilities should sum to 1.0.

Question: {question}

Available Classes:
{chr(10).join(class_descriptions)}

Return only the JSON object, no other text:
"""
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a classification expert. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            probabilities = json.loads(content)
            
            # Normalize probabilities to sum to 1.0
            total = sum(probabilities.values())
            if total > 0:
                probabilities = {k: v/total for k, v in probabilities.items()}
            
            return probabilities
            
        except Exception as e:
            print(f"LLM matching error: {e}")
            # Fallback: equal distribution
            equal_prob = 1.0 / len(classes)
            return {cls['class_id']: equal_prob for cls in classes}