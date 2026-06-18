"""Comment tab charts: word cloud and bar chart (matplotlib-based)."""
import base64
import html
import io
import logging
import os
import random
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

logger = logging.getLogger(__name__)


def render_wordcloud(word_frequencies: Dict[str, int], save_path: Optional[str] = None) -> str:
    """
    Render a word cloud as a base64-embedded image.

    Uses matplotlib WordCloud library to generate a professional visualization
    with brand colors, matching the style of the old QAQC scripts.

    Args:
        word_frequencies: Dictionary of {word: count}
        save_path: Optional file path to save the chart PNG (in addition to embedding)

    Returns:
        HTML string with embedded base64 image, or fallback text-based cloud
    """
    if not word_frequencies:
        logger.debug("No word frequencies provided for wordcloud")
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun texte de commentaire disponible.\" "
            "data-i18n-en=\"No comment text available.\">"
            "Aucun texte de commentaire disponible.</div>"
        )

    logger.debug(f"Generating wordcloud with {len(word_frequencies)} words, WORDCLOUD_AVAILABLE={WORDCLOUD_AVAILABLE}")

    if WORDCLOUD_AVAILABLE:
        try:
            brand_colors = ['#00AEC7', '#6BC643']

            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return random.choice(brand_colors)

            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                color_func=color_func,
                scale=2,
                max_words=100,
                min_font_size=10,
            ).generate_from_frequencies(word_frequencies)

            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(
                'Wordmap showing your most popular comments',
                fontsize=12, fontstyle='italic', pad=10
            )

            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
                    logger.info(f"WordCloud chart saved to: {save_path}")
                except Exception as save_err:
                    logger.warning(f"Failed to save wordcloud to file: {save_err}")

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode('ascii')
            buf.close()

            logger.info(f"WordCloud image generated successfully ({len(img_base64)} bytes base64)")

            return (
                f'<div class="wordcloud-image">'
                f'<img src="data:image/png;base64,{img_base64}" '
                f'alt="Word Cloud" style="max-width:100%; height:auto; border-radius:8px;" />'
                f'</div>'
            )

        except Exception as e:
            logger.warning(f"WordCloud image generation failed, falling back to text: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    logger.info(f"Using fallback text-based word cloud (WORDCLOUD_AVAILABLE={WORDCLOUD_AVAILABLE})")
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:40]
    if not sorted_words:
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucun texte de commentaire disponible.\" "
            "data-i18n-en=\"No comment text available.\">"
            "Aucun texte de commentaire disponible.</div>"
        )

    counts = [c for _, c in sorted_words]
    min_count = min(counts)
    max_count = max(counts)
    span_html = []
    for word, count in sorted_words:
        escaped_word = html.escape(word)
        if max_count == min_count:
            size = 18
        else:
            size = 12 + (count - min_count) / (max_count - min_count) * 22
        span_html.append(
            f"<span class=\"word\" style=\"font-size:{size:.0f}px\">{escaped_word}</span>"
        )
    return "<div class=\"wordcloud\">" + " ".join(span_html) + "</div>"


def render_comment_bar_chart(
    total_intervals: int,
    intervals_with_comments: int,
    avg_length: float = 0,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a matplotlib bar chart for comment statistics as a base64-embedded image.

    Matches the style of the old QAQC scripts' comment statistics bar chart.

    Args:
        total_intervals: Total number of logging intervals
        intervals_with_comments: Number of intervals with non-empty comments
        avg_length: Average comment length in characters
        save_path: Optional file path to save the chart PNG (in addition to embedding)

    Returns:
        HTML string with embedded base64 image
    """
    if total_intervals == 0:
        return "<div class=\"empty\">No data available for chart.</div>"

    try:
        intervals_without = total_intervals - intervals_with_comments
        comment_pct = (intervals_with_comments / total_intervals * 100) if total_intervals > 0 else 0

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        categories = ['No Comment', 'With Comment']
        counts = [intervals_without, intervals_with_comments]
        colors = ['#c9382a', '#2f7d61']

        bars = ax.bar(categories, counts, color=colors, width=0.5)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval):,}',
                    ha='center', va='bottom', fontsize=9)

        title = f'Comment Statistics\nTotal: {total_intervals:,} intervals ({comment_pct:.1f}% with comments)'
        if avg_length > 0:
            title += f'\nAvg length: {avg_length:.0f} chars'
        ax.set_title(title, fontsize=11, fontstyle='italic', pad=10)
        ax.set_ylabel('Count', fontsize=9, fontweight='bold')

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
                logger.info(f"Comment bar chart saved to: {save_path}")
            except Exception as save_err:
                logger.warning(f"Failed to save comment chart to file: {save_err}")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('ascii')
        buf.close()

        return (
            f'<div class="comment-chart-image">'
            f'<img src="data:image/png;base64,{img_base64}" '
            f'alt="Comment Statistics" style="max-width:100%; height:auto;" />'
            f'</div>'
        )

    except Exception as e:
        logger.warning(f"Comment bar chart generation failed: {e}")
        return "<div class=\"empty\">Chart generation failed.</div>"
