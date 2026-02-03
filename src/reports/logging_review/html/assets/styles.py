"""CSS for logging review HTML report."""

CSS_STYLES: str = """        :root {
            --bg: #f6f7f8;
            --panel: #ffffff;
            --ink: #1d2025;
            --muted: #5d6672;
            --accent: #0d5b88;
            --accent-warm: #c9802a;
            --border: #d8dde3;
            --success: #2f7d61;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "Georgia", serif;
            color: var(--ink);
            background: var(--bg);
        }
        .app {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 240px;
            background: #101820;
            color: #ffffff;
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .logo {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .logo img {
            max-width: 140px;
            height: auto;
        }
        .lang-toggle {
            display: flex;
            gap: 8px;
        }
        .lang-toggle button {
            background: transparent;
            border: 1px solid #ffffff;
            color: #ffffff;
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
        }
        .lang-toggle button.active {
            background: #ffffff;
            color: #101820;
        }
        .tab-button {
            width: 100%;
            text-align: left;
            background: #1b2733;
            color: #ffffff;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
        }
        .tab-button.active {
            background: var(--accent);
        }
        .logging-detail-subnav {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        .logging-detail-subtab {
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
        }
        .logging-detail-subtab:hover {
            background: var(--bg);
        }
        .logging-detail-subtab.active {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }
        .logging-detail-subpanels {
            position: relative;
        }
        .logging-detail-subpanel {
            display: none;
        }
        .logging-detail-subpanel.active {
            display: block;
        }
        .content {
            flex: 1;
            padding: 24px;
        }
        .overview-hero {
            text-align: center;
            margin-bottom: 28px;
            padding: 24px 16px;
            border-bottom: 2px solid var(--border);
        }
        .overview-hero-title {
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .overview-hero-logger {
            margin: 0 0 4px 0;
            font-size: 20px;
            font-weight: 600;
            color: var(--accent);
        }
        .overview-hero-date {
            margin: 0;
            font-size: 16px;
            color: var(--muted);
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .panel-header h2 {
            margin: 0;
            font-size: 22px;
        }
        .header-meta {
            display: flex;
            gap: 16px;
            color: var(--muted);
            font-size: 13px;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .kpi-card {
            background: var(--panel);
            padding: 14px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        .kpi-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 20px;
            font-weight: 700;
            margin-top: 6px;
        }
        .two-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }
        .overview-two-panel {
            min-height: 65vh;
            align-items: stretch;
        }
        .overview-two-panel .panel-card {
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .overview-two-panel .map-container {
            flex: 1;
            min-height: 400px;
        }
        .overview-two-panel .map-container-leaflet {
            min-height: 400px;
        }
        .overview-two-panel .map-leaflet-viewport {
            min-height: 380px;
        }
        .overview-two-panel .plotly-chart {
            flex: 1;
            min-height: 400px;
        }
        .panel-card {
            background: var(--panel);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 16px;
        }
        .panel-card h3 {
            margin-top: 0;
            font-size: 16px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .comparison-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
        }
        .comparison-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .comparison-row {
            display: grid;
            grid-template-columns: 1fr 2fr auto;
            gap: 8px;
            align-items: center;
            margin-bottom: 6px;
        }
        .comparison-bar {
            background: #e5eaef;
            height: 8px;
            border-radius: 6px;
            overflow: hidden;
        }
        .comparison-fill {
            height: 8px;
            background: var(--accent);
        }
        .comparison-status {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .comparison-status.ok {
            background: #e2f3ec;
            color: var(--success);
        }
        .comparison-status.review {
            background: #fce8d6;
            color: var(--accent-warm);
        }
        .stat-line {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
        }
        .info-box {
            margin-top: 16px;
            background: #eef4f8;
            border-left: 4px solid var(--accent);
            padding: 12px 16px;
            border-radius: 8px;
        }
        .info-box ul {
            margin: 8px 0 0 18px;
        }
        .notes-box {
            margin-top: 16px;
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 8px;
        }
        .notes-box textarea {
            width: 100%;
            min-height: 80px;
            border: 1px solid var(--border);
            padding: 8px;
            border-radius: 6px;
            font-family: inherit;
        }
        .wordcloud-image {
            text-align: center;
            margin: 16px 0;
        }
        .wordcloud-image img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .wordcloud {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .wordcloud .word {
            background: #f0f4f7;
            padding: 4px 8px;
            border-radius: 6px;
        }
        .bar-track {
            background: #e5eaef;
            height: 10px;
            border-radius: 8px;
            overflow: hidden;
        }
        .bar-fill {
            height: 10px;
            background: var(--accent);
        }
        .bar-block {
            margin-top: 12px;
        }
        .interval-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 13px;
        }
        .interval-table th, .interval-table td {
            padding: 8px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }
        .interval-table th {
            background: #f3f5f7;
        }
        .image-cell img {
            width: 120px;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        .fines-image-cell img.fines-image, .fines-image {
            width: 280px;
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        .fines-placeholder {
            width: 280px;
            min-height: 140px;
        }
        .image-placeholder-small {
            width: 80px;
            height: 50px;
            min-width: 80px;
            min-height: 50px;
            background: #f0f2f4;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 11px;
            border-radius: 4px;
        }
        .image-placeholder {
            width: 120px;
            height: 80px;
            background: #f0f2f4;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 12px;
            border-radius: 6px;
        }
        /* Rotated images - 90 degrees to save horizontal space */
        .rotated-image-cell {
            padding: 4px !important;
            text-align: center;
        }
        .rotated-image-cell img.rotated-image {
            transform: rotate(90deg);
            width: 100px;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid var(--border);
        }
        /* Geochem table - compact tabulated display */
        .geochem-table {
            border-collapse: collapse;
            font-size: 11px;
            margin: 0;
        }
        .geochem-table tr {
            border: none;
        }
        .geochem-table td {
            padding: 2px 6px;
            border: none;
        }
        .geochem-label {
            font-weight: 600;
            color: var(--muted);
            text-align: right;
        }
        .geochem-value {
            text-align: left;
            font-family: monospace;
        }
        .geochem-na {
            color: var(--muted);
        }
        .most-likely-cell {
            font-weight: 600;
            color: var(--accent);
        }
        .strat-bar-chart { margin-top: 8px; }
        .strat-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
        .strat-code { min-width: 80px; }
        .strat-bar-track { flex: 1; background: #e5eaef; height: 14px; border-radius: 8px; overflow: hidden; }
        .strat-bar-fill { height: 14px; background: var(--accent); border-radius: 8px; }
        .strat-bar-fill.team-fill { background: #94a3b8; }
        .panel-card.full-width { width: 100%; margin-bottom: 16px; }
        .strat-count { min-width: 36px; text-align: right; }
        .comment-bar-chart { margin-top: 12px; }
        .comment-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
        .comment-bar-label { min-width: 140px; }
        .comment-bar-value { min-width: 40px; }
        .bar-fill-red { background: #c9382a !important; }
        .bar-fill-green { background: var(--success) !important; }
        .comment-chart-image { text-align: center; margin: 12px 0; }
        .comment-chart-image img { max-width: 100%; height: auto; border-radius: 8px; }
        .attribution-table { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 8px; }
        .attribution-table th, .attribution-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }
        .attribution-table th { background: #f3f5f7; }
        .grouping-group { margin-bottom: 16px; border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: var(--panel); }
        .grouping-group.cv-low { border-left: 4px solid #22c55e; }
        .grouping-group.cv-medium { border-left: 4px solid #f59e0b; }
        .grouping-group.cv-high { border-left: 4px solid #ef4444; background: #fef2f2; }
        .group-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
        .group-info { display: flex; align-items: center; gap: 12px; }
        .group-strat { font-weight: 600; font-size: 14px; color: var(--text); }
        .group-count { font-size: 12px; color: var(--muted); background: #f1f5f9; padding: 2px 8px; border-radius: 12px; }
        .cv-indicator { display: flex; align-items: center; gap: 8px; }
        .cv-label { font-size: 12px; color: var(--muted); }
        .cv-bar-container { width: 80px; height: 8px; background: #e5eaef; border-radius: 4px; overflow: hidden; }
        .cv-bar { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
        .cv-value { font-weight: 600; font-size: 13px; min-width: 40px; }
        .grouping-table { margin-top: 0; }
        .strat-cell { font-weight: 500; }
        .geochem-cell { font-size: 12px; color: var(--muted); }
        .reason-cell { max-width: 280px; font-size: 12px; }
        .plotly-chart { min-height: 260px; }
        .mineral-mismatch-stats { margin: 8px 0 12px 0; font-size: 14px; color: var(--muted); }
        .mineral-mismatch-stats .stats-note { font-style: italic; margin-left: 8px; }
        .significance-low { background: #e8f4ea; color: #1b5e20; }
        .significance-high { background: #ffebee; color: #b71c1c; }
        .charts-panel .panel-card { min-width: 320px; }
        /* Compact table styles for fines and evidence tables */
        .compact-table th, .compact-table td {
            padding: 4px 6px !important;
            font-size: 12px;
        }
        .compact-row td {
            vertical-align: middle;
        }
        .nowrap {
            white-space: nowrap;
        }
        .classification-cell {
            font-weight: 600;
            background: #fef3c7;
            border-radius: 4px;
            padding: 2px 6px !important;
        }
        .classification-cell.logged {
            background: #fee2e2;
            color: #991b1b;
        }
        .classification-cell.assay {
            background: #dcfce7;
            color: #166534;
        }
        .issue-cell {
            max-width: 180px;
            font-size: 11px;
        }
        /* Inline geochem display */
        .geochem-inline {
            display: inline-block;
            margin-right: 8px;
            font-size: 11px;
        }
        .geochem-inline b {
            color: var(--muted);
            margin-right: 2px;
        }
        .geochem-inline-cell {
            white-space: nowrap;
            font-size: 11px;
        }
        /* Expandable images – thumbnail in cell; expanded = full-resolution overlay */
        .expandable-img {
            cursor: zoom-in;
            transition: all 0.3s ease;
        }
        .expandable-img.expanded {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) rotate(90deg) !important;
            width: auto !important;
            height: auto !important;
            max-width: 90vw !important;
            max-height: 90vh !important;
            min-width: 400px !important;
            min-height: 300px !important;
            z-index: 9999;
            cursor: zoom-out;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            border-radius: 8px;
            object-fit: contain;
        }
        .image-cell-compact {
            width: 80px;
            padding: 2px !important;
        }
        .image-cell-compact img {
            width: 80px;
            height: 50px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid var(--border);
            transform: none;
        }
        /* Evidence section */
        .evidence-section {
            margin-top: 20px;
            padding: 16px;
            background: #fef9e7;
            border: 1px solid #f9e79f;
            border-radius: 8px;
        }
        .evidence-section h3 {
            margin-top: 0;
            color: #9a7b1a;
            font-size: 14px;
        }
        /* Warning box for missing data */
        .warning-box {
            padding: 16px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-left: 4px solid #f97316;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .warning-box h4 {
            margin: 0 0 8px 0;
            color: #c2410c;
            font-size: 14px;
        }
        .warning-box p {
            margin: 0;
            color: #9a3412;
            font-size: 13px;
        }
        /* Rules table for zonation */
        .rules-box {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
        }
        .rules-table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 13px;
        }
        .rules-table th, .rules-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e0f2fe;
        }
        .rules-table th {
            background: #e0f2fe;
            font-weight: 600;
        }
        .rules-table tr:nth-child(even) {
            background: #f0f9ff;
        }
        .rules-note {
            font-size: 11px;
            color: #0369a1;
            margin: 8px 0 0 0;
            font-style: italic;
        }
        .evidence-table {
            background: white;
        }
        .evidence-table th {
            background: #fef3c7 !important;
        }
        .flag-pill {
            display: inline-block;
            padding: 2px 8px;
            margin: 1px 2px 1px 0;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }
        .flag-pill-extra {
            display: inline-block;
            padding: 2px 6px;
            margin: 1px 2px 1px 0;
            font-size: 12px;
            color: var(--muted);
        }
        .outlier-modal {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .outlier-modal[aria-hidden="false"] {
            display: flex;
        }
        .outlier-modal-backdrop {
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.4);
        }
        .outlier-modal-content {
            position: relative;
            background: var(--panel);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            max-width: 480px;
            width: 90%;
            max-height: 80vh;
            overflow: auto;
        }
        .outlier-modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
        }
        .outlier-modal-header h3 {
            margin: 0;
            font-size: 18px;
        }
        .outlier-modal-close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--muted);
        }
        .outlier-modal-body {
            padding: 18px;
        }
        .outlier-modal-body p {
            margin: 0 0 8px 0;
            font-size: 14px;
        }
        .outlier-detail-geochem {
            margin-top: 12px;
            font-size: 13px;
        }
        /* Sortable table styles */
        .sortable-table th.sortable {
            cursor: pointer;
            user-select: none;
        }
        .sortable-table th.sortable:hover {
            background: #fde68a !important;
        }
        .sortable-table th.sort-asc::after { content: " ▲"; }
        .sortable-table th.sort-desc::after { content: " ▼"; }
        /* Assay suggestion conditional formatting */
        .assay-mineralised {
            background-color: #dcfce7 !important;
            color: #166534;
        }
        .assay-leached {
            background-color: #fef9c3 !important;
            color: #854d0e;
        }
        .assay-unmineralised {
            background-color: #fee2e2 !important;
            color: #991b1b;
        }
        /* Validation status formatting */
        .validation-match {
            background-color: #dcfce7 !important;
            color: #166534;
            font-weight: 500;
        }
        .validation-mismatch {
            background-color: #fee2e2 !important;
            color: #991b1b;
            font-weight: 500;
        }
        .validation-pending {
            background-color: #f3f4f6 !important;
            color: #6b7280;
        }
        .geochem-cell {
            text-align: right;
            font-family: monospace;
        }
        /* Credit text */
        .credit-text {
            font-size: 10px;
            color: #64748b;
            text-align: center;
            margin-top: 4px;
        }
        /* Print button and styles */
        .print-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .print-btn:hover {
            background: #1d6856;
        }
        @media print {
            .sidebar, .print-btn, .notes-box, .map-controls, .leaflet-control-container {
                display: none !important;
            }
            .app {
                flex-direction: column;
            }
            .content {
                width: 100%;
                padding: 0;
            }
            .tab-panel {
                display: block !important;
                page-break-after: always;
                border-bottom: 2px solid #ccc;
                padding-bottom: 20px;
                margin-bottom: 20px;
            }
            .tab-panel:last-child {
                page-break-after: avoid;
            }
            .expandable-img.expanded {
                position: relative !important;
                transform: rotate(90deg) !important;
            }
        }
        /* Map container – Leaflet fills widget */
        .map-container {
            position: relative;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #f0f4f8;
        }
        .map-container-leaflet {
            display: flex;
            flex-direction: column;
            min-height: 300px;
        }
        .map-leaflet-viewport {
            width: 100%;
            flex: 1;
            min-height: 300px;
            z-index: 0;
        }
        .map-leaflet-viewport.leaflet-container {
            font-family: inherit;
        }
        .map-legend {
            display: flex;
            gap: 16px;
            padding: 8px 12px;
            background: white;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--muted);
        }
        .map-legend-leaflet {
            flex-shrink: 0;
        }
        .legend-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .legend-dot.logger-dot {
            background: var(--accent);
            border: 1px solid white;
            box-shadow: 0 0 0 1px var(--accent);
        }
        .legend-dot.team-dot {
            background: #94a3b8;
            opacity: 0.6;
        }
        .legend {
            display: flex;
            gap: 16px;
            margin-top: 10px;
            font-size: 12px;
            color: var(--muted);
        }
        .legend .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .legend .dot.logger { background: var(--accent); }
        .legend .dot.team { background: #cbd2d9; }
        .empty {
            color: var(--muted);
            font-size: 13px;
            padding: 12px;
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }
        .summary-list {
            margin: 8px 0 0 18px;
        }
        .intervals-section {
            margin-top: 16px;
        }
        @media (max-width: 860px) {
            .app {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
                flex-direction: row;
                overflow-x: auto;
                padding: 12px;
            }
            .tab-button {
                white-space: nowrap;
            }
        }
"""
