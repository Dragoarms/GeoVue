"""
Internationalization (i18n) - French and English bilingual support.

Usage:
    from geovue_capture.gui.i18n import t, set_language

    set_language("fr")
    print(t("app_title"))  # "GeoVue Capture"
    print(t("btn_home"))   # "Accueil"
"""

from typing import Dict

# Current language
_current_lang = "en"

# Translation dictionary: key -> {lang: text}
_STRINGS: Dict[str, Dict[str, str]] = {
    # Application
    "app_title": {
        "en": "GeoVue Capture",
        "fr": "GeoVue Capture",
    },
    "app_version": {
        "en": "Version 2.8.0",
        "fr": "Version 2.8.0",
    },

    # Menu / Top bar
    "menu_file": {
        "en": "File",
        "fr": "Fichier",
    },
    "menu_settings": {
        "en": "Settings",
        "fr": "Paramètres",
    },
    "menu_language": {
        "en": "Language",
        "fr": "Langue",
    },
    "menu_help": {
        "en": "Help",
        "fr": "Aide",
    },
    "menu_quit": {
        "en": "Quit",
        "fr": "Quitter",
    },

    # Data entry
    "lbl_hole_id": {
        "en": "Hole ID",
        "fr": "ID du trou",
    },
    "lbl_depth_range": {
        "en": "Depth Range",
        "fr": "Intervalle de profondeur",
    },
    "lbl_depth_from": {
        "en": "From (m)",
        "fr": "De (m)",
    },
    "lbl_depth_to": {
        "en": "To (m)",
        "fr": "À (m)",
    },
    "lbl_depth_increment": {
        "en": "Increment (m)",
        "fr": "Incrément (m)",
    },
    "lbl_moisture": {
        "en": "Moisture",
        "fr": "Humidité",
    },
    "moisture_dry": {
        "en": "Dry",
        "fr": "Sec",
    },
    "moisture_wet": {
        "en": "Wet",
        "fr": "Humide",
    },
    "moisture_mixed": {
        "en": "Mixed",
        "fr": "Mixte",
    },

    # Stage controls
    "lbl_stage_control": {
        "en": "Stage Control",
        "fr": "Contrôle du plateau",
    },
    "btn_left": {
        "en": "Left",
        "fr": "Gauche",
    },
    "btn_right": {
        "en": "Right",
        "fr": "Droite",
    },
    "btn_home": {
        "en": "Home",
        "fr": "Accueil",
    },
    "btn_stop": {
        "en": "STOP",
        "fr": "ARRÊT",
    },
    "lbl_speed": {
        "en": "Speed",
        "fr": "Vitesse",
    },
    "lbl_position": {
        "en": "Position",
        "fr": "Position",
    },

    # Capture
    "btn_capture": {
        "en": "Capture",
        "fr": "Capturer",
    },
    "btn_next_tray": {
        "en": "Next Tray",
        "fr": "Bac suivant",
    },
    "lbl_capture_sequence": {
        "en": "Capture Sequence",
        "fr": "Séquence de capture",
    },

    # Status / Safety
    "status_ready": {
        "en": "Ready",
        "fr": "Prêt",
    },
    "status_moving": {
        "en": "Moving...",
        "fr": "En mouvement...",
    },
    "status_homing": {
        "en": "Homing...",
        "fr": "Retour à l'origine...",
    },
    "status_stopped": {
        "en": "Stopped",
        "fr": "Arrêté",
    },
    "status_limit_home": {
        "en": "LIMIT: Home switch triggered",
        "fr": "LIMITE : Fin de course accueil déclenché",
    },
    "status_limit_far": {
        "en": "LIMIT: Far switch triggered",
        "fr": "LIMITE : Fin de course éloigné déclenché",
    },
    "status_limit_cleared": {
        "en": "Limit switches clear",
        "fr": "Fins de course libérés",
    },
    "status_emergency_stop": {
        "en": "EMERGENCY STOP",
        "fr": "ARRÊT D'URGENCE",
    },

    # Processed list
    "lbl_processed_today": {
        "en": "Processed Today",
        "fr": "Traités aujourd'hui",
    },
    "col_time": {
        "en": "Time",
        "fr": "Heure",
    },
    "col_hole_id": {
        "en": "Hole ID",
        "fr": "ID trou",
    },
    "col_depth": {
        "en": "Depth",
        "fr": "Profondeur",
    },
    "col_moisture": {
        "en": "Moisture",
        "fr": "Humidité",
    },

    # Dialogs
    "dlg_confirm_delete": {
        "en": "Move this image to the deleted folder?",
        "fr": "Déplacer cette image dans le dossier supprimé ?",
    },
    "dlg_delete_title": {
        "en": "Delete Image",
        "fr": "Supprimer l'image",
    },
    "dlg_input_error": {
        "en": "Input Error",
        "fr": "Erreur de saisie",
    },
    "dlg_enter_hole_id": {
        "en": "Please enter Hole ID",
        "fr": "Veuillez saisir l'ID du trou",
    },
    "dlg_select_depth": {
        "en": "Please select a depth range",
        "fr": "Veuillez sélectionner un intervalle de profondeur",
    },
    "dlg_confirm_quit": {
        "en": "Are you sure you want to quit?",
        "fr": "Êtes-vous sûr de vouloir quitter ?",
    },
    "dlg_quit_title": {
        "en": "Quit GeoVue Capture",
        "fr": "Quitter GeoVue Capture",
    },

    # Platform
    "lbl_platform": {
        "en": "Platform",
        "fr": "Plateforme",
    },
    "platform_leadscrew": {
        "en": "Leadscrew (DM556)",
        "fr": "Vis à billes (DM556)",
    },
    "platform_pilbara": {
        "en": "Pilbara (TMC2225)",
        "fr": "Pilbara (TMC2225)",
    },

    # Diagnostics
    "lbl_diagnostics": {
        "en": "Diagnostics",
        "fr": "Diagnostics",
    },
    "lbl_switch_status": {
        "en": "Switch Status",
        "fr": "État des fins de course",
    },
    "lbl_home_switch": {
        "en": "Home Switch",
        "fr": "Fin de course accueil",
    },
    "lbl_far_switch": {
        "en": "Far Switch",
        "fr": "Fin de course éloigné",
    },
    "switch_open": {
        "en": "Open (OK)",
        "fr": "Ouvert (OK)",
    },
    "switch_triggered": {
        "en": "TRIGGERED",
        "fr": "DÉCLENCHÉ",
    },
    "lbl_motor_state": {
        "en": "Motor State",
        "fr": "État du moteur",
    },
}


def set_language(lang: str):
    """Set the active language ('en' or 'fr')."""
    global _current_lang
    if lang in ("en", "fr"):
        _current_lang = lang
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'en' or 'fr'.")


def get_language() -> str:
    """Get the current language code."""
    return _current_lang


def t(key: str) -> str:
    """
    Translate a string key to the current language.

    Falls back to English if key exists but current language is missing.
    Returns the key itself if not found at all.
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(_current_lang, entry.get("en", key))


def get_all_keys() -> list:
    """Return all available translation keys."""
    return list(_STRINGS.keys())
