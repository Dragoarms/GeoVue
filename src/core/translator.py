# /core/translator.py

"""
Handles translations for the application.
Loads translations from CSV files and provides functionality to translate strings.
"""

import os
import csv
import logging
import platform
import re
from typing import Dict, Optional, Any, Set, Tuple
from difflib import SequenceMatcher

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
        self.enable_missing_translation_csv_log = True  # Toggle manually in code
        self._warned_missing = set()
        self._logged_missing_keys = set()  # Prevent duplicates across runs
        self.missing_translation_log_path = None
        

        # Fuzzy matching configuration

        self.enable_fuzzy_matching = True
        self.fuzzy_threshold = 0.85  # Similarity threshold (0.0 to 1.0)
        self._normalized_cache = {}  # Cache for normalized keys
        self._fuzzy_match_cache = {}  # Cache fuzzy match results


        if self.enable_missing_translation_csv_log:
            appdata = os.getenv('APPDATA') or os.path.expanduser('~')
            self.missing_translation_log_path = os.path.join(appdata, 'GeoVue', 'missing_translations.csv')

    
            # Load existing logged keys to prevent duplicates
    
            if self.missing_translation_log_path and os.path.exists(self.missing_translation_log_path):
                try:
                    with open(self.missing_translation_log_path, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        next(reader)  # Skip header
                        for row in reader:
                            if row:
                                self._logged_missing_keys.add(row[0])
                    logger.info(f"Loaded {len(self._logged_missing_keys)} existing missing translations")
                except Exception as e:
                    logger.warning(f"Failed to load existing missing translations: {e}")
    

            # Ensure CSV exists with header
            if self.missing_translation_log_path and not os.path.exists(self.missing_translation_log_path):
                os.makedirs(os.path.dirname(self.missing_translation_log_path), exist_ok=True)
                with open(self.missing_translation_log_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Key'])

        
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
        

        # Build normalized translation index for fuzzy matching

        self._build_normalized_index()


        # Set current language
        self.detect_system_language()

    # ===================================================
    # Normalization and fuzzy matching methods
    # ===================================================
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for fuzzy matching by:
        - Converting to lowercase
        - Removing extra whitespace
        - Removing common punctuation (but not emojis or special characters)
        - Standardizing newlines
        - Preserving emojis and special Unicode characters
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Cache normalized results
        if text in self._normalized_cache:
            return self._normalized_cache[text]
            
        normalized = text.lower()
        
        # Standardize whitespace and newlines
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\n+', ' ', normalized)
        
        # Only remove specific ASCII punctuation marks
        # This preserves emojis, accented characters, and other Unicode
        # Keep: letters, numbers, spaces, placeholders {}, and all non-ASCII characters
        punctuation_to_remove = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`|~]'
        
        # But preserve placeholders like {variable}
        # First, temporarily replace placeholders with a safe token
        import uuid
        placeholder_token = str(uuid.uuid4())
        placeholders = re.findall(r'\{[^}]*\}', normalized)
        for i, placeholder in enumerate(placeholders):
            normalized = normalized.replace(placeholder, f"{placeholder_token}_{i}")
        
        # Remove punctuation
        normalized = re.sub(punctuation_to_remove, '', normalized)
        
        # Restore placeholders
        for i, placeholder in enumerate(placeholders):
            normalized = normalized.replace(f"{placeholder_token}_{i}", placeholder)
        
        # Trim
        normalized = normalized.strip()
        
        self._normalized_cache[text] = normalized
        return normalized
    
    def _build_normalized_index(self):
        """Build an index of normalized translations for faster fuzzy matching."""
        self._normalized_index = {}
        
        for lang_code, translations in self.translations.items():
            self._normalized_index[lang_code] = {}
            for key in translations.keys():
                normalized = self._normalize_text(key)
                if normalized not in self._normalized_index[lang_code]:
                    self._normalized_index[lang_code][normalized] = []
                self._normalized_index[lang_code][normalized].append(key)
    
    def _find_fuzzy_match(self, text: str, language: str) -> Optional[Tuple[str, float]]:
        """
        Find the best fuzzy match for the given text.
        
        Args:
            text: Text to find match for
            language: Language code to search in
            
        Returns:
            Tuple of (matched_key, similarity_score) or None
        """
        if not self.enable_fuzzy_matching:
            return None
            
        # Check cache first
        cache_key = (text, language)
        if cache_key in self._fuzzy_match_cache:
            return self._fuzzy_match_cache[cache_key]
            
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return None
            
        translations = self.translations.get(language, {})
        if not translations:
            return None
            
        best_match = None
        best_score = 0.0
        
        # First try exact normalized match
        normalized_index = self._normalized_index.get(language, {})
        if normalized_text in normalized_index:
            # Get the first exact normalized match
            exact_key = normalized_index[normalized_text][0]
            self._fuzzy_match_cache[cache_key] = (exact_key, 1.0)
            return (exact_key, 1.0)
        
        # Try fuzzy matching on all keys
        for key in translations.keys():
            normalized_key = self._normalize_text(key)
            
            # Use SequenceMatcher for similarity calculation
            similarity = SequenceMatcher(None, normalized_text, normalized_key).ratio()
            
            if similarity > best_score and similarity >= self.fuzzy_threshold:
                best_match = key
                best_score = similarity
        
        result = (best_match, best_score) if best_match else None
        self._fuzzy_match_cache[cache_key] = result

        return result
    # ===================================================
    def _match_placeholders(self, text: str, language: str) -> Optional[str]:
        """Attempt to match text against translation keys containing placeholders.

        This allows calls like ``translate("Download for Windows")`` to match a
        translation key such as ``"Download for {system}"`` and substitute the
        captured value back into the translated string.

        Args:
            text: The full text to match, including variable values.
            language: The target language code.

        Returns:
            The formatted translation if a placeholder pattern matches, else
            ``None``.
        """
        translations = self.translations.get(language, {})
        for key, translation in translations.items():
            placeholders = re.findall(r"{([^}]+)}", key)
            if not placeholders:
                continue

            # Build regex pattern from key, converting placeholders to named groups
            pattern = re.escape(key)
            for name in placeholders:
                pattern = pattern.replace(re.escape(f"{{{name}}}"), fr"(?P<{name}>.+)")
            match = re.fullmatch(pattern, text)
            if match:
                try:
                    return translation.format(**match.groupdict())
                except Exception as e:
                    logger.warning(f"Placeholder substitution failed for '{text}': {e}")
                    return translation

        return None
    # ===================================================

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
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
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
        #     # Force French for testing purposes.
        # self.current_language = "fr"
        # return True
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
    
            # Clear fuzzy match cache when language changes
    
            self._fuzzy_match_cache.clear()
    
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
        

        # Enhanced translation lookup with fuzzy matching

        translated_text = None
        fuzzy_matched = False
        
        # First try exact match
        if text in translations:
            translated_text = translations[text]
        else:
            # Try fuzzy matching
            fuzzy_result = self._find_fuzzy_match(text, self.current_language)
            if fuzzy_result:
                matched_key, score = fuzzy_result
                translated_text = translations[matched_key]
                fuzzy_matched = True
                logger.debug(f"Fuzzy matched '{text}' to '{matched_key}' (score: {score:.2f})")
            else:
                # Try matching against keys with placeholders
                placeholder_result = self._match_placeholders(text, self.current_language)
                if placeholder_result is not None:
                    translated_text = placeholder_result
        
        # If still no translation found
        if translated_text is None:
            # Check if we should log this as missing
            normalized_text = self._normalize_text(text)
            should_log = True
            
            # Check if a similar text was already logged
            for logged_key in self._logged_missing_keys:
                if self._normalize_text(logged_key) == normalized_text:
                    should_log = False
                    break
            
            if should_log and text not in self._logged_missing_keys:
                if text not in self._warned_missing:
                    logger.debug(f"Missing Translation: '{text}'")
                    self._warned_missing.add(text)

                    if self.enable_missing_translation_csv_log and self.missing_translation_log_path:
                        try:
                            with open(self.missing_translation_log_path, 'a', newline='', encoding='utf-8-sig') as f:
                                writer = csv.writer(f)
                                writer.writerow([text])
                            self._logged_missing_keys.add(text)
                        except Exception as e:
                            logger.warning(f"Failed to write missing translation to CSV: {e}")

            translated_text = text

        
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
        
        # Check exact match first
        if key in translations:
            return True
            
        # Check fuzzy match
        if self.enable_fuzzy_matching:
            fuzzy_result = self._find_fuzzy_match(key, self.current_language)
            return fuzzy_result is not None
            
        return False
