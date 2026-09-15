
"""
Intent taxonomy for the AppleSupport Hiver agent.
"""

INTENTS = [
    "ios_update_issue",
    "app_crash_or_freeze",
    "device_performance",
    "battery_issue",
    "connectivity_issue",
    "media_playback_issue",
    "authentication_or_code",
    "hardware_repair_or_damage",
    "system_ui_keyboard_or_notification_bug",
    "icloud_photos_or_mail",
    "messaging_or_communication",
    "information_or_how_to",
]

INTENT_DEFINITIONS = {
    "ios_update_issue":
        "Problems caused by or related to updating iOS, update failures, update prompts, or behavior immediately after an iOS update.",

    "app_crash_or_freeze":
        "Apps crashing, freezing, closing unexpectedly, or repeatedly becoming unresponsive.",

    "device_performance":
        "General device slowness, lag, poor responsiveness, or performance degradation not primarily caused by one app.",

    "battery_issue":
        "Battery draining quickly, charging problems, poor battery life, or battery-related concerns.",

    "connectivity_issue":
        "Wi-Fi, Bluetooth, mobile/network connection, internet connectivity, or network settings problems.",

    "media_playback_issue":
        "Apple Music, music, video, audio, or other media playback problems.",

    "authentication_or_code":
        "Apple ID sign-in, password, verification code, activation, authentication, or account access problems.",

    "hardware_repair_or_damage":
        "Physical device damage, cracked/broken screens, hardware failures, or device boot/startup problems.",

    "system_ui_keyboard_or_notification_bug":
        "Keyboard, notifications, interface/UI behavior, or system-level UI bugs.",

    "icloud_photos_or_mail":
        "iCloud, Photos, Mail synchronization or access problems.",

    "messaging_or_communication":
        "iMessage, SMS, text messaging, or sending/receiving communication problems.",

    "information_or_how_to":
        "Requests asking how to perform a task, configure a feature, or obtain general product information.",
}

LABEL_TO_ID = {i: i for i in INTENTS}
ID_TO_LABEL = {i: i for i in INTENTS}
