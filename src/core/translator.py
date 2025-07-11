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
from typing import Tuple, Union, Dict, List, Optional, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

from core.file_manager import FileManager


class TranslationManager:
    def __init__(
        self,
        file_manager: Optional["FileManager"] = None,
        config: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        csv_path: Optional[str] = None,
        translations_dict: Optional[Dict[str, Dict[str, str]]] = None,
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
            appdata = os.getenv("APPDATA") or os.path.expanduser("~")
            self.missing_translation_log_path = os.path.join(
                appdata, "GeoVue", "missing_translations.csv"
            )

            # Load existing logged keys to prevent duplicates

            if self.missing_translation_log_path and os.path.exists(
                self.missing_translation_log_path
            ):
                try:
                    with open(
                        self.missing_translation_log_path, "r", encoding="utf-8-sig"
                    ) as f:
                        reader = csv.reader(f)
                        next(reader)  # Skip header
                        for row in reader:
                            if row:
                                self._logged_missing_keys.add(row[0])
                    logger.info(
                        f"Loaded {len(self._logged_missing_keys)} existing missing translations"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load existing missing translations: {e}")

            # Ensure CSV exists with header
            if self.missing_translation_log_path and not os.path.exists(
                self.missing_translation_log_path
            ):
                os.makedirs(
                    os.path.dirname(self.missing_translation_log_path), exist_ok=True
                )
                with open(
                    self.missing_translation_log_path,
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["Key"])

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

        # Cache for all translation keys
        self._all_translation_keys = None

    def _get_all_translation_keys(self):
        if self._all_translation_keys is None:
            self._all_translation_keys = set()
            for lang_translations in self.translations.values():
                self._all_translation_keys.update(lang_translations.keys())
        return self._all_translation_keys

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
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"\n+", " ", normalized)

        # Only remove specific ASCII punctuation marks
        # This preserves emojis, accented characters, and other Unicode
        # Keep: letters, numbers, spaces, placeholders {}, and all non-ASCII characters
        punctuation_to_remove = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`|~]'

        # But preserve placeholders like {variable}
        # First, temporarily replace placeholders with a safe token
        import uuid

        placeholder_token = str(uuid.uuid4())
        placeholders = re.findall(r"\{[^}]*\}", normalized)
        for i, placeholder in enumerate(placeholders):
            normalized = normalized.replace(placeholder, f"{placeholder_token}_{i}")

        # Remove punctuation
        normalized = re.sub(punctuation_to_remove, "", normalized)

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

    def _find_fuzzy_match(
        self, text: str, language: str
    ) -> Optional[Tuple[str, float]]:
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

    def get_current_language(self):
        """Return the currently selected language."""
        return self.current_language

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
            with open(self.csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                headers = next(reader)

                # First column is the key
                key_index = 0

                # Other columns are languages
                language_indices = {
                    header: i for i, header in enumerate(headers) if i > 0
                }

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

            if system == "Windows":
                # Set and get the current locale
                locale.setlocale(locale.LC_ALL, "")
                language_code = locale.getlocale()
                if language_code:  # language_code is a tuple, (language, encoding)
                    # Extract primary language
                    lang = (
                        language_code[0].split("_")[0].lower()
                        if language_code[0]
                        else None
                    )
                    if lang and lang in self.translations:
                        self.current_language = lang
                        return True
        except Exception as e:
            print(f"Error detecting system language: {e}")

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
            "fr": "Français",
            # Add more language names as needed
        }
        return language_names.get(language_code, language_code)

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

            # Clear caches when language changes
            self._fuzzy_match_cache.clear()
            self._all_translation_keys = None  # Reset the keys cache

            return True
        return False

    def _extract_template_and_values(
        self, text: str, translation_keys: set
    ) -> Tuple[Optional[str], Optional[Union[tuple, dict]]]:
        """
        Try to match a formatted string back to its template and extract the values.

        Args:
            text: The potentially formatted string (e.g., "Processing 5 files")
            translation_keys: Set of available translation keys

        Returns:
            Tuple of (template_key, extracted_values) or (None, None) if no match
        """
        # For each template key that contains placeholders
        candidates = []

        for template_key in translation_keys:
            if "{" not in template_key:
                continue

            # Build a regex pattern from the template
            # First, escape special regex characters except for placeholders
            escaped_template = re.escape(template_key)

            # Extract placeholder names and positions
            placeholder_pattern = r"\\\{([^}]*)\\\}"
            placeholders = re.findall(placeholder_pattern, escaped_template)

            if not placeholders:
                continue

            # Replace escaped placeholders with capture groups
            regex_pattern = escaped_template

            # For each placeholder, use appropriate regex based on context
            for i, placeholder in enumerate(placeholders):
                # Determine appropriate regex based on placeholder name
                capture_pattern = r"(.+?)"  # Default: non-greedy capture

                if placeholder:  # Named placeholder
                    # Use more specific patterns based on common placeholder names
                    placeholder_lower = placeholder.lower()
                    if any(
                        term in placeholder_lower
                        for term in ["count", "number", "num", "total", "size"]
                    ):
                        capture_pattern = r"(\d+)"  # Numbers only
                    elif any(
                        term in placeholder_lower for term in ["id", "code", "ref"]
                    ):
                        capture_pattern = r"([A-Z0-9\-_]+)"  # Alphanumeric IDs
                    elif any(
                        term in placeholder_lower
                        for term in ["path", "file", "folder", "dir"]
                    ):
                        capture_pattern = r"([^\n]+)"  # Paths (anything except newline)
                    elif any(term in placeholder_lower for term in ["date", "time"]):
                        capture_pattern = r"([^\n]+)"  # Dates/times (flexible)

                    regex_pattern = regex_pattern.replace(
                        f"\\{{{placeholder}\\}}", capture_pattern
                    )
                else:  # Positional placeholder {}
                    regex_pattern = regex_pattern.replace(r"\{\}", capture_pattern)

            # Try to match the input text against this pattern
            regex_pattern = f"^{regex_pattern}$"

            try:
                match = re.match(regex_pattern, text, re.IGNORECASE)

                if match:
                    # Calculate match quality (prefer exact case matches)
                    quality_score = 1.0
                    if template_key.lower() != text.lower():
                        # Check how much of the template is literal vs placeholders
                        literal_chars = len(re.sub(r"\{[^}]*\}", "", template_key))
                        total_chars = len(template_key)
                        if total_chars > 0:
                            quality_score = literal_chars / total_chars

                    # Extract the captured values
                    captured_values = match.groups()

                    # If we have named placeholders, create a kwargs dict
                    if placeholders and all(p for p in placeholders):
                        extracted_kwargs = {}
                        for j, placeholder in enumerate(placeholders):
                            if j < len(captured_values):
                                extracted_kwargs[placeholder] = captured_values[j]
                        candidates.append(
                            (quality_score, template_key, extracted_kwargs)
                        )
                    else:
                        # Return as positional args
                        candidates.append(
                            (quality_score, template_key, captured_values)
                        )
            except re.error:
                # Invalid regex, skip this template
                continue

        # Return the best match (highest quality score)
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            _, best_template, best_values = candidates[0]
            return best_template, best_values

        return None, None

    def translate(self, text, *args, **kwargs):
        """
        Translate text to the current language with support for variable substitution.
        Now with intelligent detection of pre-formatted strings.

        Args:
            text: Text to translate (may contain {} placeholders or be pre-formatted)
            *args: Positional arguments for string formatting
            **kwargs: Keyword arguments for string formatting

        Returns:
            Translated and formatted text or original text if no translation found

        Examples:
            translate("Hello {}", "World")  # -> "Bonjour World" (if French)
            translate("Processing {count} files", count=5)  # -> "Traitement de 5 fichiers"
            translate("Processing 5 files")  # -> "Traitement de 5 fichiers" (auto-detected)
            translate("{hole_id} - {depth_from}-{depth_to}m", hole_id="AB1234", depth_from=10, depth_to=20)
        """
        # Handle None or empty string
        if text is None or text == "":
            return text

        # Get translation dictionary for current language
        translations = self.translations.get(self.current_language, {})

        # Try to detect if this is a pre-formatted string
        # Only attempt this if no args/kwargs provided and no placeholders in text
        if not args and not kwargs and "{" not in text:
            # IMPORTANT: Only get keys from the SOURCE language (English)
            # We want to match against English templates to translate to current language
            english_keys = set(self.translations.get("en", {}).keys())

            # Try to extract template and values
            template_key, extracted_values = self._extract_template_and_values(
                text, english_keys
            )

            if template_key and extracted_values:
                # We found a matching template!
                logger.debug(
                    f"Detected pre-formatted string: '{text}' matches template '{template_key}'"
                )

                # Now translate the template with the extracted values
                if isinstance(extracted_values, dict):
                    return self.translate(template_key, **extracted_values)
                else:
                    return self.translate(template_key, *extracted_values)

        # Check if this is already the correct translation for current language
        # This prevents infinite loops when the source and target are the same
        if self.current_language == "en" and text in translations:
            # For English, if it's a valid key, translate it
            pass  # Continue to translation
        elif text in translations.values():
            # If it's already a translated value in the current language, just format and return
            try:
                if args or kwargs:
                    return text.format(*args, **kwargs)
                else:
                    return text
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(
                    f"Translation formatting failed for already-translated text '{text}': {e}"
                )
                return text

        # Enhanced translation lookup with fuzzy matching
        translated_text = None
        fuzzy_matched = False

        # First try exact match
        if text in translations:
            translated_text = translations[text]
            logger.debug("Translated: %s -> %s", text, translated_text)
        else:
            # Try fuzzy matching
            fuzzy_result = self._find_fuzzy_match(text, self.current_language)
            if fuzzy_result:
                matched_key, score = fuzzy_result
                translated_text = translations[matched_key]
                fuzzy_matched = True
                logger.debug(
                    f"Fuzzy matched '{text}' to '{matched_key}' (score: {score:.2f})"
                )

        # If still no translation found
        if translated_text is None:
            # Before logging as missing, check if this text is a translation in another language
            # This helps identify when already-translated text from another language is passed
            is_translation_from_other_language = False
            for lang_code, lang_translations in self.translations.items():
                if (
                    lang_code != self.current_language
                    and text in lang_translations.values()
                ):
                    is_translation_from_other_language = True
                    logger.debug(
                        f"Text '{text}' appears to be a {lang_code} translation"
                    )
                    break

            if not is_translation_from_other_language:
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

                        if (
                            self.enable_missing_translation_csv_log
                            and self.missing_translation_log_path
                        ):
                            try:
                                with open(
                                    self.missing_translation_log_path,
                                    "a",
                                    newline="",
                                    encoding="utf-8-sig",
                                ) as f:
                                    writer = csv.writer(f)
                                    writer.writerow([text])
                                self._logged_missing_keys.add(text)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to write missing translation to CSV: {e}"
                                )

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
