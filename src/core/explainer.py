import re
import numpy as np

class ExplainabilityEngine:
    def __init__(self, classifier):
        self.clf = classifier

    def explain(self, text: str, initial_score: float, triggers: list, entities: dict) -> dict:
        """
        Generates an explanation for the scam verdict.
        Returns a list of 'Highlights' (word, type, impact_score).
        """
        words = text.split()
        highlights = []
        
        # 1. Static Highlights (Rules & NER)
        # We prefer these as they are "Hard" evidence
        
        # Highlight Triggers
        for pattern_str in triggers:
            # Re-compile to find span
            try:
                for match in re.finditer(pattern_str, text, re.IGNORECASE):
                    highlights.append({
                        "span": match.span(),
                        "word": match.group(),
                        "type": "TRIGGER",
                        "impact": 1.0 # Max impact implication
                    })
            except:
                pass

        # Highlight Entities (NER)
        for ent_type, vals in entities.items():
            for val in vals:
                # Simple string search for the entity value in text
                # In real app, we'd use span from Natasha, but we passed normalized strings
                # So we do best-effort search here or pass spans from NER usually
                start = text.lower().find(val.lower()) # Basic find
                if start != -1:
                    highlights.append({
                        "span": (start, start + len(val)),
                        "word": text[start:start+len(val)],
                        "type": f"NER_{ent_type}",
                        "impact": 0.8
                    })

        # 2. Dynamic Impact Analysis (Simplified LIME/Occlusion)
        # We mask each word and check how much the ML score DROPS.
        # This identifies words that "fooled" or "convinced" the AI.
        
        # Only do this if ML Score is high enough to matter
        if initial_score > 0.4: 
            base_score = initial_score
            
            # Optimization: check 3-grams or individual significant words to save time?
            # For short messages, word-by-word is fine.
            
            clean_words = [w.strip(".,!?:") for w in words]
            
            for i, word in enumerate(clean_words):
                if len(word) < 3: continue 
                
                # Create perturbed text (remove word)
                perturbed_text = text.replace(word, "", 1)
                
                # Predict (Fast mode, no extensive checks)
                res = self.clf.predict(perturbed_text, strict_mode=True) # strict to skip extra checks
                new_score = res['ml_score']
                
                drop = base_score - new_score
                
                # If removing the word dropped the score significantly (>0.1), it's important
                if drop > 0.05:
                    highlights.append({
                        "span": None, # Complex to track span after replace
                        "word": word,
                        "type": "ML_FACTOR",
                        "impact": round(drop, 3)
                    })

        # Format output
        # Deduplicate and prioritize
        unique_highlights = {}
        for h in highlights:
            key = h['word'].lower()
            if key not in unique_highlights or h['impact'] > unique_highlights[key]['impact']:
                unique_highlights[key] = h
                
        return list(unique_highlights.values())

    def visualize(self, text, highlights):
        """Returns console-friendly visualized text"""
        vis_text = text
        # Sort highlights by length desc to avoid replacing substrings incorrectly
        sorted_h = sorted(highlights, key=lambda x: len(x['word']), reverse=True)
        
        for h in sorted_h:
            word = h['word']
            tag = h['type']
            # Color codes for console
            # RED for Trigger, YELLOW for ML, CYAN for NER
            replacement = f"[{word}]"
            
            if "TRIGGER" in tag:
                replacement = f"🔴[{word}]"
            elif "NER" in tag:
                replacement = f"🔵[{word}]"
            elif "ML" in tag:
                 replacement = f"⚠️[{word}]"
                 
            vis_text = vis_text.replace(word, replacement)
            
        return vis_text
