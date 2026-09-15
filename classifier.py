
"""
DistilBERT intent classifier.

Important:
The label order below MUST match the trained model's output IDs.
"""

from pathlib import Path
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# EXACT output order from the trained DistilBERT checkpoint
MODEL_LABELS = [
    "app_crash_or_freeze",
    "authentication_or_code",
    "battery_issue",
    "connectivity_issue",
    "device_performance",
    "hardware_repair_or_damage",
    "icloud_photos_or_mail",
    "information_or_how_to",
    "ios_update_issue",
    "media_playback_issue",
    "messaging_or_communication",
    "system_ui_keyboard_or_notification_bug",
]


class IntentClassifier:

    def __init__(self, model_dir=None):

        if model_dir is None:
            model_dir = (
                Path(__file__).resolve().parent
                / "models"
                / "distilbert_intent_classifier"
            )

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir)
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # Use the model's own ID mapping when available.
        config_mapping = self.model.config.id2label

        labels_from_config = [
            config_mapping[i]
            for i in range(self.model.config.num_labels)
        ]

        # Verify that the checkpoint agrees with our expected order.
        if labels_from_config != MODEL_LABELS:
            raise ValueError(
                "Model label mapping does not match the repository "
                "MODEL_LABELS order.\n"
                f"Model: {labels_from_config}\n"
                f"Expected: {MODEL_LABELS}"
            )

        self.labels = MODEL_LABELS

    def _rule_override(self, text):

        text_l = text.lower()

        # ----------------------------------------------------
        # Hardware
        # ----------------------------------------------------
        if (
            re.search(r"\b(macbook|mac)\b", text_l)
            and re.search(
                r"\b(won't boot|wont boot|doesn't boot|"
                r"doesnt boot|not turning on|won't turn on|"
                r"wont turn on)\b",
                text_l,
            )
        ):
            return "hardware_repair_or_damage", 0.99

        if (
            re.search(
                r"\b(cracked|broken|damaged)\b",
                text_l
            )
            and re.search(
                r"\b(screen|iphone|ipad|macbook|device)\b",
                text_l,
            )
        ):
            return "hardware_repair_or_damage", 0.99

        # ----------------------------------------------------
        # Battery
        # ----------------------------------------------------
        if re.search(
            r"\b(battery|charging|charge|battery life|"
            r"draining|drain quickly|battery health)\b",
            text_l,
        ):
            return "battery_issue", 0.99

        # ----------------------------------------------------
        # Connectivity
        # ----------------------------------------------------
        if re.search(
            r"\b(wifi|wi-fi|bluetooth|network|internet|"
            r"connection|connectivity|reset network settings)\b",
            text_l,
        ):
            return "connectivity_issue", 0.99

        # ----------------------------------------------------
        # App crash
        # ----------------------------------------------------
        if re.search(
            r"\b(crash|crashes|crashing|freeze|freezes|"
            r"freezing|force close|closes unexpectedly)\b",
            text_l,
        ):
            return "app_crash_or_freeze", 0.99

        # ----------------------------------------------------
        # Authentication
        # ----------------------------------------------------
        if re.search(
            r"\b(apple id|appleid|sign in|signin|password|"
            r"verification code|verification|activation)\b",
            text_l,
        ):
            return "authentication_or_code", 0.99

        # ----------------------------------------------------
        # Messaging
        # ----------------------------------------------------
        if re.search(
            r"\b(imessage|messages|text messages|sms|"
            r"send messages|receive messages)\b",
            text_l,
        ):
            return "messaging_or_communication", 0.99

        # ----------------------------------------------------
        # Media
        # ----------------------------------------------------
        if re.search(
            r"\b(apple music|music player|music|video|"
            r"playback|audio playback)\b",
            text_l,
        ):
            return "media_playback_issue", 0.99

        return None

    @torch.no_grad()
    def predict(self, text):

        text = str(text).strip()

        if not text:
            return {
                "intent": "information_or_how_to",
                "confidence": 0.0,
                "source": "empty_input",
            }

        override = self._rule_override(text)

        if override is not None:

            intent, confidence = override

            return {
                "intent": intent,
                "confidence": confidence,
                "source": "rule_override",
            }

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        index = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[index].item()
        )

        intent = self.labels[index]

        return {
            "intent": intent,
            "confidence": confidence,
            "source": "distilbert",
        }
