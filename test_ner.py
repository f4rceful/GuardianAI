from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)
from src.core.ner import EntityExtractor # Старый класс

def test_ner_comparison():
    print("--- Инициализация Natasha ---")
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    
    print("--- Инициализация старого NER ---")
    old_ner = EntityExtractor()
    
    test_texts = [
        "Майор Волков из ФСБ требует перевода на счет в Сбербанке.",
        "Это служба безопасности Сбербанка, меня зовут Елена.",
        "Генерал Браун из Сирии хочет отправить посылку в Москву.",
        "Срочно переведи деньги маме на карту Тинькофф.",
        "Я выиграл автомобиль Лада Веста в лотерее Авторадио.",
        "Ваша карта заблокирована. Позвоните в службу безопасности: +79001234567"
    ]
    
    print("\n" + "="*60)
    print(f"{'Text':<50} | {'Old Regex':<20} | {'New Natasha':<20}")
    print("="*60)
    
    for text in test_texts:
        # 1. Old Regex
        old_res = old_ner.extract(text)
        old_str = str(old_res.get('AUTHORITY', []) + old_res.get('FINANCE', []))
        
        # 2. Natasha
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        
        # Normalize
        for span in doc.spans:
            span.normalize(morph_vocab)
            
        natasha_str = ", ".join([f"{s.type}:{s.normal}" for s in doc.spans])
        
        print(f"{text[:45]}... | {old_str[:20]:<20} | {natasha_str}")
        print("-" * 60)

if __name__ == "__main__":
    test_ner_comparison()
