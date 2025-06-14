# /core/translator.py

"""
Handles translations for the application.
Loads translations from CSV files and provides functionality to translate strings.
"""

import os
import csv
import logging
import platform
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

from core.file_manager import FileManager



class TranslationManager:
    def __init__(
        self, 
        file_manager: Optional['FileManager'] = None,
        config: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        csv_path: Optional[str] = None,
        translations_dict: Optional[Dict[str, Dict[str, str]]] = None
    ) -> None:
        """
        Initialize the TranslationManager with specified configuration.
        
        Args:
            file_manager: FileManager instance for path resolution.
            config: Configuration dictionary.
            language: Override for default language.
            csv_path: Override for path to translations.csv.
            translations_dict: Preloaded translations dictionary (for testing or override).
        """
        self.file_manager = file_manager
        self.config = config or {}
        self.current_language = language or self.config.get("language", "en")
        
        # Simple, direct translations path resolution
        if csv_path:
            # Direct path provided - highest priority
            self.csv_path = csv_path
            logger.info(f"Using provided translations path: {self.csv_path}")
        elif self.config.get("translations_path") and self.file_manager:
            # Resolve config path against base_dir
            config_path = self.config.get("translations_path")
            self.csv_path = os.path.join(self.file_manager.base_dir, config_path)
            logger.info(f"Using config-based translations path: {self.csv_path}")
        else:
            # No path available
            self.csv_path = None
            logger.warning("No translations path specified")
        
        # Load translations
        if translations_dict is not None:
            self.translations = translations_dict
            logger.info("Using provided translations dictionary")
        else:
            self.translations = self._load_translations_from_csv()
        

        # Set current language
        self.detect_system_language()


    def get_current_language(self):
        """Return the currently selected language."""
        return self.current_language

    def set_language(self, language_code):
        """Set the current language."""
        self.current_language = language_code
    
    def _load_translations_from_csv(self) -> Dict[str, Dict[str, str]]:
        """
        Load translations from CSV file.
        
        Returns:
            Dictionary mapping language codes to dictionaries of translations.
        """
        translations: Dict[str, Dict[str, str]] = {}
        
        try:
            # Check if we have a path
            if not self.csv_path:
                logger.warning("No translations path specified")
                return {}
                
            # Check if file exists
            if not os.path.exists(self.csv_path):
                logger.warning(f"Translations file not found at: {self.csv_path}")
                return {}
                
            logger.info(f"Loading translations from: {self.csv_path}")
            
            # Read the CSV file
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # First column is the key
                key_index = 0
                
                # Other columns are languages
                language_indices = {header: i for i, header in enumerate(headers) if i > 0}
                
                # Initialize languages
                for lang in language_indices:
                    translations[lang] = {}
                    
                # Read translations
                for row in reader:
                    if not row:
                        continue
                        
                    key = row[key_index]
                    for lang, index in language_indices.items():
                        if index < len(row) and row[index]:
                            translations[lang][key] = row[index]
            
            return translations
            
        except Exception as e:
            logger.error(f"Error loading translations: {e}")
            return {}

    def detect_system_language(self):
        """Detect system language from locale settings."""
        try:
            import locale
            # Get system platform
            system = platform.system()

            if system == 'Windows':
                # Set and get the current locale
                locale.setlocale(locale.LC_ALL, '')
                language_code = locale.getlocale()
                if language_code:  # language_code is a tuple, (language, encoding)
                    # Extract primary language
                    lang = language_code[0].split('_')[0].lower() if language_code[0] else None
                    if lang and lang in self.translations:
                        self.current_language = lang
                        return True
            elif system in ('Darwin', 'Linux'):
                # Try to get locale from environment variables
                lang_env = os.environ.get('LANG', '')
                if lang_env:
                    lang = lang_env.split('_')[0].lower()
                    if lang in self.translations:
                        self.current_language = lang
                        return True
        except Exception as e:
            print(f"Error detecting system language: {e}")

        return False
    
    def set_language(self, language_code):
        """
        Set the current language.
        
        Args:
            language_code: Language code (e.g., "en", "fr")
            
        Returns:
            bool: True if language was set successfully, False otherwise
        """
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_available_languages(self):
        """
        Get list of available languages.
        
        Returns:
            List of language codes
        """
        return list(self.translations.keys())
    
    def get_language_name(self, language_code):
        """Get human-readable name for a language code."""
        language_names = {
            "en": "English",
            "fr": "Français"
            # Add more language names as needed
        }
        return language_names.get(language_code, language_code)
    
    def translate(self, text, *args, **kwargs):
        """
        Translate text to the current language with support for variable substitution.
        
        Args:
            text: Text to translate (may contain {} placeholders)
            *args: Positional arguments for string formatting
            **kwargs: Keyword arguments for string formatting
            
        Returns:
            Translated and formatted text or original text if no translation found
            
        Examples:
            translate("Hello {}", "World")  # -> "Bonjour World" (if French)
            translate("Processing {count} files", count=5)  # -> "Traitement de 5 fichiers"
            translate("{hole_id} - {depth_from}-{depth_to}m", hole_id="AB1234", depth_from=10, depth_to=20)
        """
        # Handle None or empty string
        if text is None or text == "":
            return text
            
        # Get translation dictionary for current language
        translations = self.translations.get(self.current_language, {})
        
        # Get the base translation (without variables substituted)
        translated_text = translations.get(text, text)
        
        # Apply variable substitution if args or kwargs provided
        try:
            if args or kwargs:
                return translated_text.format(*args, **kwargs)
            else:
                return translated_text
        except (KeyError, ValueError, IndexError) as e:
            # If formatting fails, log the error and return the unformatted translation
            logger.warning(f"Translation formatting failed for '{text}': {e}")
            return translated_text

    def t(self, text, *args, **kwargs):
        """Shorthand method for translate with variable support."""
        return self.translate(text, *args, **kwargs)

    def has_translation(self, key: str) -> bool:
        """
        Check if the given key has a translation for the current language.
        
        Args:
            key: The translation key to check
            
        Returns:
            True if translation exists, False otherwise
        """
        if not key:
            return False
            
        # Get translations for current language
        translations = self.translations.get(self.current_language, {})
        
        # Check if key exists
        has_key = key in translations
        
        return has_key
