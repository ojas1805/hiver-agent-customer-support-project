import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import AppleSupportAgent

def main():
    if not os.getenv('GROQ_API_KEY'):
        raise RuntimeError(
            'GROQ_API_KEY is not set.'
        )

    print('=' * 60)
    print('HIVER APPLE SUPPORT AGENT')
    print('=' * 60)

    message = input('\nCustomer message: ').strip()

    if not message:
        print('No message provided.')
        return

    agent = AppleSupportAgent()
    result = agent.run(message)

    print('\nIntent:')
    print(result['intent'])

    print('\nIntent confidence:')
    print(round(result['intent_confidence'], 4))

    print('\nRetrieval score:')
    print(round(result['retrieval_top_score'], 4))

    print('\nRetrieval strength:')
    print(result['retrieval_strength'])

    print('\nDraft reply:')
    print(result['reply'])

    print('\nEscalate:')
    print(result['escalate'])

    print('\nEscalation reason:')
    print(result['escalation_reason'])

if __name__ == '__main__':
    main()
