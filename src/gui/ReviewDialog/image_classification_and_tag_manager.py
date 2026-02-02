# src/gui/ReviewDialog/image_classification_and_tag_manager.py
"""
Unified manager for image classifications and tags.
Classifications are mutually exclusive (only one per image).
Tags are additive (multiple tags per image).
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ItemDefinition:
    """Base definition for classifications and tags"""

    id: str  # Internal ID (e.g., "biff", "gold_assay")
    label: str  # Display name (e.g., "BIFf", "Gold Assay")
    color: str  # Hex color (e.g., "#4CAF50")
    keybinding: Optional[str] = None  # Keyboard shortcut (e.g., "1", "g")
    is_active: bool = True  # Whether button is visible
    is_default: bool = False  # Whether this is a built-in item
    order: int = 0  # Display order
    icon: Optional[str] = None  # Optional emoji/icon (e.g., "⚠️", "🏆")
    item_type: str = "classification"  # "classification" or "tag"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ItemDefinition":
        """Create from dictionary"""
        return cls(**data)


class ImageClassificationAndTagManager:
    """Manages both classifications (mutually exclusive) and tags (additive)"""

    # Default classifications
    DEFAULT_CLASSIFICATIONS = [
        ItemDefinition(
            id="biff",
            label="BIFf",
            color="#4CAF50",
            keybinding="1",
            is_active=True,
            is_default=True,
            order=0,
            item_type="classification",
        ),
        ItemDefinition(
            id="biff_s",
            label="BIFf-s",
            color="#9ACD32",
            keybinding="2",
            is_active=True,
            is_default=True,
            order=1,
            item_type="classification",
        ),
        ItemDefinition(
            id="compact",
            label="Compact",
            color="#FF9800",
            keybinding="3",
            is_active=True,
            is_default=True,
            order=2,
            item_type="classification",
        ),
        ItemDefinition(
            id="bifhm",
            label="BIFhm",
            color="#f44336",
            keybinding="4",
            is_active=True,
            is_default=True,
            order=3,
            item_type="classification",
        ),
        ItemDefinition(
            id="not_confident",
            label="Not Confident",
            color="#000000",
            keybinding="5",
            is_active=True,
            is_default=True,
            order=4,
            item_type="classification",
        ),
        ItemDefinition(
            id="other",
            label="Other",
            color="#2196F3",
            keybinding="6",
            is_active=True,
            is_default=True,
            order=5,
            item_type="classification",
        ),
    ]

    # Default tags
    DEFAULT_TAGS = [
        ItemDefinition(
            id="gold_assays",
            label="Gold Assays",
            color="#FFD700",  # Gold color
            keybinding="g",
            is_active=True,
            is_default=True,
            order=0,
            item_type="tag",
            icon="⭐",
        ),
        ItemDefinition(
            id="jasperlitic",
            label="Jasperlitic",
            color="#8B0000",  # Dark red
            keybinding="j",
            is_active=True,
            is_default=True,
            order=1,
            item_type="tag",
            icon="🧱",
        ),
    ]

    def __init__(self, config_manager=None):
        """
        Initialize unified manager

        Args:
            config_manager: Optional ConfigManager for persistence
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.items: Dict[str, ItemDefinition] = {}

        # Load from config or use defaults
        self._load_items()

    def _load_items(self):
        """Load all items from config or initialize defaults"""
        if self.config_manager:
            # Load classifications
            saved_classifications = self.config_manager.get("classifications", None)
            if saved_classifications:
                for item_data in saved_classifications:
                    try:
                        item_data["item_type"] = "classification"
                        item = ItemDefinition.from_dict(item_data)
                        self.items[item.id] = item
                    except Exception as e:
                        self.logger.error(
                            f"Error loading classification {item_data}: {e}"
                        )
            else:
                # No saved classifications - load defaults
                for item in self.DEFAULT_CLASSIFICATIONS:
                    self.items[item.id] = item

            # Load tags
            saved_tags = self.config_manager.get("tags", None)
            if saved_tags:
                for item_data in saved_tags:
                    try:
                        item_data["item_type"] = "tag"
                        item = ItemDefinition.from_dict(item_data)
                        self.items[item.id] = item
                    except Exception as e:
                        self.logger.error(f"Error loading tag {item_data}: {e}")
            else:
                # No saved tags - load defaults
                for item in self.DEFAULT_TAGS:
                    self.items[item.id] = item

            # If still nothing loaded, something is wrong
            if not self.items:
                self.logger.warning("No items loaded - initializing defaults")
                self._initialize_defaults()
            
            # Always save after loading to ensure defaults are persisted
            self.save_items()
        else:
            # No config manager - just use defaults
            self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize with default classifications and tags"""
        for item in self.DEFAULT_CLASSIFICATIONS + self.DEFAULT_TAGS:
            self.items[item.id] = item

    def save_items(self):
        """Save all items to config (separate keys for classifications and tags)"""
        if self.config_manager:
            # Separate classifications and tags
            classifications = [
                item.to_dict()
                for item in sorted(
                    [i for i in self.items.values() if i.item_type == "classification"],
                    key=lambda x: x.order,
                )
            ]
            tags = [
                item.to_dict()
                for item in sorted(
                    [i for i in self.items.values() if i.item_type == "tag"],
                    key=lambda x: x.order,
                )
            ]

            self.config_manager.set("classifications", classifications)
            self.config_manager.set("tags", tags)
            self.logger.info(
                f"Saved {len(classifications)} classifications and {len(tags)} tags to config"
            )

    # === Generic item methods ===

    def get_all_items(self, item_type: Optional[str] = None) -> List[ItemDefinition]:
        """Get all items, optionally filtered by type"""
        items = self.items.values()
        if item_type:
            items = [i for i in items if i.item_type == item_type]
        return sorted(items, key=lambda x: x.order)

    def get_active_items(self, item_type: Optional[str] = None) -> List[ItemDefinition]:
        """Get only active items, optionally filtered by type"""
        items = [i for i in self.items.values() if i.is_active]
        if item_type:
            items = [i for i in items if i.item_type == item_type]
        return sorted(items, key=lambda x: x.order)

    def get_item(self, item_id: str) -> Optional[ItemDefinition]:
        """Get a specific item by ID"""
        return self.items.get(item_id)

    def add_item(
        self,
        item_id: str,
        label: str,
        color: str,
        item_type: str,
        keybinding: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> ItemDefinition:
        """Add a new item (classification or tag)"""
        # Check for duplicate ID
        if item_id in self.items:
            raise ValueError(f"Item with ID '{item_id}' already exists")

        # Check for duplicate keybinding
        if keybinding:
            existing = self.get_item_by_keybinding(keybinding)
            if existing:
                raise ValueError(
                    f"Keybinding '{keybinding}' already used by '{existing.label}'"
                )

        # Create new item with next order for its type
        existing_of_type = [i for i in self.items.values() if i.item_type == item_type]
        max_order = max((i.order for i in existing_of_type), default=-1)

        item = ItemDefinition(
            id=item_id,
            label=label,
            color=color,
            keybinding=keybinding,
            icon=icon,
            is_active=True,
            is_default=False,
            order=max_order + 1,
            item_type=item_type,
        )

        self.items[item_id] = item
        self.save_items()

        self.logger.info(f"Added new {item_type}: {label} ({item_id})")
        return item

    def update_item(
        self,
        item_id: str,
        label: Optional[str] = None,
        color: Optional[str] = None,
        keybinding: Optional[str] = None,
        is_active: Optional[bool] = None,
        icon: Optional[str] = None,
    ):
        """Update an existing item"""
        if item_id not in self.items:
            raise ValueError(f"Item '{item_id}' not found")

        item = self.items[item_id]

        # Check for duplicate keybinding if changing
        if keybinding is not None and keybinding != item.keybinding:
            existing = self.get_item_by_keybinding(keybinding)
            if existing and existing.id != item_id:
                raise ValueError(
                    f"Keybinding '{keybinding}' already used by '{existing.label}'"
                )

        # Update fields
        if label is not None:
            item.label = label
        if color is not None:
            item.color = color
        if keybinding is not None:
            item.keybinding = keybinding
        if is_active is not None:
            item.is_active = is_active
        if icon is not None:
            item.icon = icon

        self.save_items()
        self.logger.info(f"Updated {item.item_type}: {item.label} ({item_id})")

    def delete_item(self, item_id: str):
        """Delete an item (only if not default)"""
        if item_id not in self.items:
            raise ValueError(f"Item '{item_id}' not found")

        item = self.items[item_id]

        if item.is_default:
            raise ValueError(f"Cannot delete default {item.item_type}s")

        del self.items[item_id]
        self.save_items()
        self.logger.info(f"Deleted {item.item_type}: {item.label} ({item_id})")

    def get_item_by_keybinding(self, keybinding: str) -> Optional[ItemDefinition]:
        """Find item by keybinding"""
        for item in self.items.values():
            if item.keybinding == keybinding:
                return item
        return None

    def get_color_for_item(self, item_id: str) -> str:
        """Get color for an item - supports both ID and label lookup"""
        # First try direct ID lookup
        if item_id in self.items:
            return self.items[item_id].color
        
        # Fallback: try case-insensitive ID lookup
        item_id_lower = item_id.lower()
        if item_id_lower in self.items:
            return self.items[item_id_lower].color
        
        # Fallback: search by label (case-insensitive)
        for item in self.items.values():
            if item.label.lower() == item_id.lower():
                return item.color
        
        # Not found - return default gray
        return "#666666"

    def reorder_items(self, ordered_ids: List[str], item_type: str):
        """Reorder items of a specific type"""
        for index, item_id in enumerate(ordered_ids):
            if item_id in self.items and self.items[item_id].item_type == item_type:
                self.items[item_id].order = index

        self.save_items()
        self.logger.info(f"Reordered {item_type}s")

    # === Convenience methods for backward compatibility ===

    def get_all_classifications(self) -> List[ItemDefinition]:
        """Get all classifications"""
        return self.get_all_items("classification")

    def get_active_classifications(self) -> List[ItemDefinition]:
        """Get active classifications"""
        return self.get_active_items("classification")

    def get_classification(self, class_id: str) -> Optional[ItemDefinition]:
        """Get a classification by ID"""
        item = self.get_item(class_id)
        return item if item and item.item_type == "classification" else None

    def get_all_tags(self) -> List[ItemDefinition]:
        """Get all tags"""
        return self.get_all_items("tag")

    def get_active_tags(self) -> List[ItemDefinition]:
        """Get active tags"""
        return self.get_active_items("tag")

    def get_tag(self, tag_id: str) -> Optional[ItemDefinition]:
        """Get a tag by ID"""
        item = self.get_item(tag_id)
        return item if item and item.item_type == "tag" else None

    def get_column_display_name(self, column_name: str) -> str:
        """Get user-friendly display name for a column
        
        Args:
            column_name: Column name (e.g., 'tag_gold_assay', 'consensus_classification')
            
        Returns:
            User-friendly display name
        """
        # Handle tag columns
        if column_name.startswith("tag_"):
            tag_id = column_name[4:]
            tag_def = self.get_tag(tag_id)
            if tag_def:
                # Use icon + label if icon exists, otherwise just label
                if tag_def.icon:
                    return f"{tag_def.icon} {tag_def.label}"
                return tag_def.label
            return column_name
        
        # Map other computed fields
        display_names = {
            "consensus_classification": "Consensus Classification",
            "review_count": "Review Count",
            "agreement": "Agreement",
            "hole_id": "Hole ID",
            "depth_from": "Depth From",
            "depth_to": "Depth To",
            "classification": "Classification",
            "moisture_status": "Moisture Status",
            "comments": "Comments",
        }
        
        return display_names.get(column_name, column_name)

    # === Additional backward compatibility methods for color access ===

    def get_color_for_classification(self, class_id: str) -> str:
        """Get color for a classification (backward compatibility)"""
        return self.get_color_for_item(class_id)

    def get_color_for_tag(self, tag_id: str) -> str:
        """Get color for a tag (backward compatibility)"""
        return self.get_color_for_item(tag_id)

    def add_classification(
        self,
        class_id: str,
        label: str,
        color: str,
        keybinding: Optional[str] = None,
    ) -> ItemDefinition:
        """Add a classification (backward compatibility)"""
        return self.add_item(
            item_id=class_id,
            label=label,
            color=color,
            item_type="classification",
            keybinding=keybinding,
        )

    def update_classification(
        self,
        class_id: str,
        label: Optional[str] = None,
        color: Optional[str] = None,
        keybinding: Optional[str] = None,
        is_active: Optional[bool] = None,
    ):
        """Update a classification (backward compatibility)"""
        return self.update_item(
            item_id=class_id,
            label=label,
            color=color,
            keybinding=keybinding,
            is_active=is_active,
        )

    def delete_classification(self, class_id: str):
        """Delete a classification (backward compatibility)"""
        return self.delete_item(class_id)

    def reorder_classifications(self, ordered_ids: List[str]):
        """Reorder classifications (backward compatibility)"""
        return self.reorder_items(ordered_ids, "classification")

    def add_tag(
        self,
        tag_id: str,
        label: str,
        color: str,
        keybinding: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> ItemDefinition:
        """Add a tag (backward compatibility)"""
        return self.add_item(
            item_id=tag_id,
            label=label,
            color=color,
            item_type="tag",
            keybinding=keybinding,
            icon=icon,
        )

    def update_tag(
        self,
        tag_id: str,
        label: Optional[str] = None,
        color: Optional[str] = None,
        keybinding: Optional[str] = None,
        is_active: Optional[bool] = None,
        icon: Optional[str] = None,
    ):
        """Update a tag (backward compatibility)"""
        return self.update_item(
            item_id=tag_id,
            label=label,
            color=color,
            keybinding=keybinding,
            is_active=is_active,
            icon=icon,
        )

    def delete_tag(self, tag_id: str):
        """Delete a tag (backward compatibility)"""
        return self.delete_item(tag_id)

    def reorder_tags(self, ordered_ids: List[str]):
        """Reorder tags (backward compatibility)"""
        return self.reorder_items(ordered_ids, "tag")
