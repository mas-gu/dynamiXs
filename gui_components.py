#!/usr/bin/env python3
"""
GUI Components and Style Constants for DynamiXs
Following LunaNMR UX Style Guide v0.9

This module defines all visual design constants and reusable components
for the DynamiXs NMR analysis GUI, ensuring consistency with the LunaNMR
software suite design language.
"""

import tkinter as tk
import customtkinter as ctk

# ============================================================================
# BACKGROUND COLORS
# ============================================================================

BG_COLOR = "#FAFAFA"          # Main window background - softer white
PANEL_BG_COLOR = "#F5F5F7"    # Secondary panels/frames - Apple's signature grey
FRAME_BG_COLOR = "#FFFFFF"    # Card/container backgrounds - pure white

# ============================================================================
# TEXT COLORS
# ============================================================================

PRIMARY_TEXT = "#1C1C1E"      # Primary content text (softer than pure black)
SECONDARY_TEXT = "#8E8E93"    # Secondary/help text
DISABLED_TEXT = "#C7C7CC"     # Disabled states
LABEL_TEXT = "#1C1C1E"        # Label text

# ============================================================================
# BUTTON COLORS
# ============================================================================

# Primary Action Buttons
PRIMARY_BUTTON_BG = "#5B9EE5"       # Softer, pleasant blue
PRIMARY_BUTTON_HOVER = "#4A8DD4"    # Darker blue on hover
PRIMARY_BUTTON_TEXT = "#FFFFFF"     # White text

# Secondary/Utility Buttons
SECONDARY_BUTTON_BG = "#E5E5EA"         # Light grey
SECONDARY_BUTTON_HOVER = "#D1D1D6"      # Slightly darker grey
SECONDARY_BUTTON_TEXT = "#1C1C1E"       # Near-black text
SECONDARY_BUTTON_BORDER = "#C8C8CD"     # Subtle border (prevents edge artifacts)

# Destructive Action Buttons
DESTRUCTIVE_BUTTON_BG = "#E8554E"       # Softer red (not aggressive)
DESTRUCTIVE_BUTTON_HOVER = "#D44943"    # Darker red on hover
DESTRUCTIVE_BUTTON_TEXT = "#FFFFFF"     # White text

# ============================================================================
# ACCENT COLORS
# ============================================================================

SUCCESS_GREEN = "#34C759"     # Successful operations, progress bars
SUCCESS_GREEN_SOFT = "#81C784"  # Soft green for loaded state
SUCCESS_GREEN_HOVER = "#66BB6A"  # Hover state for loaded buttons
WARNING_ORANGE = "#F0A04B"    # Warnings, attention needed
ERROR_RED = "#E8554E"         # Errors (matches destructive buttons)
INFO_BLUE = "#5B9EE5"         # Informational messages (matches primary)

# ============================================================================
# BORDER & SEPARATOR COLORS
# ============================================================================

BORDER_COLOR = "#D1D1D6"       # Light grey borders
SEPARATOR_COLOR = "#E5E5EA"    # Separators and dividers

# ============================================================================
# CORNER RADIUS
# ============================================================================

BUTTON_CORNER_RADIUS = 10     # Buttons
FRAME_CORNER_RADIUS = 12      # Panels and label frames
DIALOG_CORNER_RADIUS = 14     # Modal dialogs, popups
CARD_CORNER_RADIUS = 8        # Small UI cards/elements

# ============================================================================
# SPACING (8pt Grid System)
# ============================================================================

SPACING_XS = 2    # Tight spacing (label-to-field, within components)
SPACING_SM = 4    # Default spacing (between related elements)
SPACING_MD = 8    # Section spacing (between groups)
SPACING_LG = 12   # Major section breaks
SPACING_XL = 16   # Window padding, large separations

# ============================================================================
# FONT CONFIGURATION
# ============================================================================

FONT_FAMILY = "TkDefaultFont"
FONT_FAMILY_MONO = "Courier"

# Font sizes and weights
FONT_LARGE_HEADER = ("TkDefaultFont", 14, "bold")     # Section titles, main controls
FONT_MEDIUM_HEADER = ("TkDefaultFont", 12, "bold")    # Subsection titles
FONT_SECTION_LABEL = ("TkDefaultFont", 11, "bold")    # Important labels
FONT_BODY = ("TkDefaultFont", 10)                     # Standard UI text, buttons
FONT_SMALL = ("TkDefaultFont", 9)                     # Help text, secondary info
FONT_TINY = ("TkDefaultFont", 8)                      # Metadata, annotations
FONT_MONO = ("Courier", 10)                           # Data display, code output

# ============================================================================
# CUSTOM COMPONENTS
# ============================================================================

class CTkLabelFrame(ctk.CTkFrame):
    """
    Custom labeled frame with rounded corners matching LunaNMR style.

    This component creates a frame with an optional label at the top,
    using the standardized corner radius and color scheme.
    """

    def __init__(self, parent, text="", padding=10, corner_radius=FRAME_CORNER_RADIUS, **kwargs):
        # Extract padding
        if isinstance(padding, (tuple, list)):
            pad_x, pad_y = padding[0], padding[1] if len(padding) > 1 else padding[0]
        else:
            pad_x = pad_y = padding

        # Create frame with panel background
        super().__init__(parent, corner_radius=corner_radius,
                        fg_color=PANEL_BG_COLOR, **kwargs)

        # Add header container if text provided
        self.header = None
        if text:
            self.header = ctk.CTkFrame(self, fg_color="transparent")
            self.header.pack(fill=tk.X, padx=pad_x, pady=(pad_y, pad_y//2))

            label = ctk.CTkLabel(self.header, text=text,
                               font=FONT_SECTION_LABEL,
                               text_color=PRIMARY_TEXT)
            label.pack(side=tk.LEFT)

        # Create content container with transparent background
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=(0, pad_y))


def create_primary_button(parent, text, command, width=None, **kwargs):
    """Create a primary action button with LunaNMR styling."""
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=width if width else 100,
        font=FONT_BODY,
        corner_radius=BUTTON_CORNER_RADIUS,
        fg_color=PRIMARY_BUTTON_BG,
        hover_color=PRIMARY_BUTTON_HOVER,
        text_color=PRIMARY_BUTTON_TEXT,
        **kwargs
    )


def create_secondary_button(parent, text, command, width=None, **kwargs):
    """Create a secondary/utility button with LunaNMR styling."""
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=width if width else 80,
        font=FONT_BODY,
        corner_radius=BUTTON_CORNER_RADIUS,
        fg_color=SECONDARY_BUTTON_BG,
        hover_color=SECONDARY_BUTTON_HOVER,
        text_color=SECONDARY_BUTTON_TEXT,
        border_width=1,
        border_color=SECONDARY_BUTTON_BORDER,
        **kwargs
    )


def create_destructive_button(parent, text, command, width=None, **kwargs):
    """Create a destructive action button with LunaNMR styling."""
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=width if width else 80,
        font=FONT_BODY,
        corner_radius=BUTTON_CORNER_RADIUS,
        fg_color=DESTRUCTIVE_BUTTON_BG,
        hover_color=DESTRUCTIVE_BUTTON_HOVER,
        text_color=DESTRUCTIVE_BUTTON_TEXT,
        **kwargs
    )


def create_label(parent, text, font=FONT_BODY, **kwargs):
    """Create a label with LunaNMR styling."""
    return ctk.CTkLabel(
        parent,
        text=text,
        font=font,
        text_color=PRIMARY_TEXT,
        **kwargs
    )


def create_entry(parent, textvariable=None, width=None, **kwargs):
    """Create an entry field with LunaNMR styling."""
    return ctk.CTkEntry(
        parent,
        textvariable=textvariable,
        width=width if width else 200,
        font=FONT_BODY,
        fg_color=FRAME_BG_COLOR,
        text_color=PRIMARY_TEXT,
        border_color=BORDER_COLOR,
        **kwargs
    )


# ============================================================================
# CUSTOMTKINTER CONFIGURATION
# ============================================================================

def configure_customtkinter():
    """Configure CustomTkinter with LunaNMR appearance settings."""
    ctk.set_appearance_mode("light")  # Force light mode for consistency
    ctk.set_default_color_theme("blue")  # Base theme (we override with custom colors)


# Initialize CustomTkinter on module import
configure_customtkinter()
