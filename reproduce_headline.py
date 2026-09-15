from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier import IntentClassifier

EVAL_PATH = ROOT / 'data' / 'processed' / 'clean_eval_dataset.csv'

def main():
    print('=' * 60)
    print('REPRODUCING INTENT HEADLINE RESULT')
    print('=' * 60)

    df = pd.read_csv(EVAL_PATH)
    classifier = IntentClassifier()
    predictions = []

    for i, row in df.iterrows():
        result = classifier.predict(row['customer_text_clean'])
        predictions.append(result['intent'])
        print(f'Processed {i + 1}/{len(df)}', end='\r')

    y_true = df['evaluation_intent']
    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)

    print('\n')
    print('Accuracy:', round(accuracy, 4))
    print('Macro F1:', round(macro_f1, 4))
    print('Weighted F1:', round(weighted_f1, 4))

if __name__ == '__main__':
    main()
