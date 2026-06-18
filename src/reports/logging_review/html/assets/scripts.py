"""JavaScript for logging review HTML report."""

JS_SCRIPTS: str = """        const defaultLang = 'fr';
        const langKey = 'loggingReviewLang';

        function applyLanguage(lang) {
            document.querySelectorAll('[data-i18n-fr], [data-i18n-html-fr]').forEach(el => {
                const htmlText = el.getAttribute(`data-i18n-html-${lang}`);
                const plainText = el.getAttribute(`data-i18n-${lang}`);
                if (htmlText) {
                    el.innerHTML = htmlText;
                } else if (plainText) {
                    el.textContent = plainText;
                }
            });
            document.querySelectorAll('.lang-toggle button').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.lang === lang);
            });
            localStorage.setItem(langKey, lang);
        }

        document.querySelectorAll('.lang-toggle button').forEach(btn => {
            btn.addEventListener('click', () => applyLanguage(btn.dataset.lang));
        });

        const savedLang = localStorage.getItem(langKey) || defaultLang;
        applyLanguage(savedLang);

        function decodePlotlyAttr(str) {
            if (!str) return str;
            return str
                .replace(/&amp;/g, '&')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&quot;/g, '"')
                .replace(/&#x27;/g, "'")
                .replace(/&#39;/g, "'");
        }
        function initPlotlyCharts() {
            if (typeof Plotly === 'undefined') return;
            document.querySelectorAll('.plotly-chart').forEach(function(el) {
                var dataStr = el.getAttribute('data-plotly-data');
                var layoutStr = el.getAttribute('data-plotly-layout');
                if (!dataStr || !layoutStr) return;
                try {
                    var data = JSON.parse(decodePlotlyAttr(dataStr));
                    var layout = JSON.parse(decodePlotlyAttr(layoutStr));
                    if (Array.isArray(data) && data.length > 0) Plotly.react(el, data, layout);
                } catch (e) { console.warn('Plotly init failed', e); }
            });
        }
        function resizePlotlyInActivePanel() {
            if (typeof Plotly === 'undefined') return;
            var active = document.querySelector('.tab-panel.active');
            if (!active) return;
            var charts = active.querySelectorAll('.plotly-chart');
            setTimeout(function() {
                charts.forEach(function(el) {
                    try { if (el._fullData) Plotly.Plots.resize(el); } catch (e) {}
                });
            }, 50);
        }

        const tabs = document.querySelectorAll('.tab-panel');
        const buttons = document.querySelectorAll('.tab-button');
        function activateTab(tabId) {
            tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === tabId));
            buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.tabButton === tabId));
            resizePlotlyInActivePanel();
        }
        buttons.forEach(btn => {
            btn.addEventListener('click', () => activateTab(btn.dataset.tabButton));
        });
        const firstTab = buttons.length ? buttons[0].dataset.tabButton : 'overview';
        activateTab(firstTab);

        function activateLoggingDetailSubtab(subtabId) {
            const section = document.querySelector('.tab-panel[data-tab="logging-detail"]');
            if (!section) return;
            section.querySelectorAll('.logging-detail-subtab').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.loggingDetailSubtab === subtabId);
            });
            section.querySelectorAll('.logging-detail-subpanel').forEach(panel => {
                panel.classList.toggle('active', panel.dataset.loggingDetailSubpanel === subtabId);
            });
            resizePlotlyInActivePanel();
        }
        document.querySelectorAll('.logging-detail-subtab').forEach(btn => {
            btn.addEventListener('click', () => activateLoggingDetailSubtab(btn.dataset.loggingDetailSubtab));
        });

        function setupNotes() {
            document.querySelectorAll('[data-note-id]').forEach(area => {
                const key = `note::${area.dataset.noteId}`;
                area.value = localStorage.getItem(key) || '';
                area.addEventListener('input', () => {
                    localStorage.setItem(key, area.value);
                });
            });
        }

        function setupCheckboxes() {
            document.querySelectorAll('[data-review-id]').forEach(box => {
                const key = `review::${box.dataset.reviewId}`;
                box.checked = localStorage.getItem(key) === '1';
                box.addEventListener('change', () => {
                    localStorage.setItem(key, box.checked ? '1' : '0');
                });
            });
        }

        setupNotes();
        setupCheckboxes();

        (function setupOutlierDetailsModal() {
            const modal = document.getElementById('outlier-details-modal');
            if (!modal) return;
            const backdrop = modal.querySelector('.outlier-modal-backdrop');
            const closeBtn = modal.querySelector('.outlier-modal-close');
            const holeEl = modal.querySelector('.outlier-detail-hole');
            const depthEl = modal.querySelector('.outlier-detail-depth');
            const recordedEl = modal.querySelector('.outlier-detail-recorded');
            const likelyEl = modal.querySelector('.outlier-detail-likely');
            const reasonEl = modal.querySelector('.outlier-detail-reason');
            const geochemEl = modal.querySelector('.outlier-detail-geochem');
            function closeModal() {
                modal.setAttribute('aria-hidden', 'true');
            }
            function openModal(data) {
                holeEl.textContent = 'Hole: ' + (data.hole_id || '-');
                depthEl.textContent = 'Depth: ' + (data.depth || '-');
                recordedEl.textContent = 'Recorded as: ' + (data.recorded_as || '-');
                likelyEl.textContent = 'Most likely: ' + (data.most_likely || '-');
                reasonEl.textContent = data.reason || '-';
                let geochemHtml = '';
                if (data.geochem && typeof data.geochem === 'object') {
                    for (const [k, v] of Object.entries(data.geochem)) {
                        const val = v != null && !Number.isNaN(v) ? Number(v).toFixed(2) : '-';
                        geochemHtml += '<p>' + k + ': ' + val + '</p>';
                    }
                }
                geochemEl.innerHTML = geochemHtml || '<p>-</p>';
                modal.setAttribute('aria-hidden', 'false');
            }
            document.querySelectorAll('.outlier-details-link').forEach(function(link) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const payload = link.getAttribute('data-details');
                    if (payload) {
                        try {
                            const data = JSON.parse(decodePlotlyAttr(payload));
                            openModal(data);
                        } catch (err) { console.warn('Outlier details parse failed', err); }
                    }
                });
            });
            if (backdrop) backdrop.addEventListener('click', closeModal);
            if (closeBtn) closeBtn.addEventListener('click', closeModal);
        })();

        // Leaflet map: OpenTopoMap basemap, logger/team markers, fit bounds
        function setupLeafletMap() {
            const el = document.getElementById('leaflet-map');
            if (!el || typeof L === 'undefined') return;
            const pointsJson = el.dataset.mapPoints;
            const boundsJson = el.dataset.mapBounds;
            if (!pointsJson || pointsJson === '[]') return;
            let points, bounds;
            try {
                points = JSON.parse(pointsJson);
                bounds = boundsJson && boundsJson !== 'null' ? JSON.parse(boundsJson) : null;
            } catch (e) {
                return;
            }
            if (!points.length) return;

            const map = L.map(el, { scrollWheelZoom: true });
            L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
                maxZoom: 17,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
            }).addTo(map);

            const loggerColor = '#0d5b88';
            const teamColor = '#94a3b8';
            const loggerLayer = L.layerGroup();
            const teamLayer = L.layerGroup();
            points.forEach(function(p) {
                const lat = p.lat;
                const lng = p.lng;
                const holeId = p.hole_id || '';
                const isLogger = p.is_logger;
                const marker = L.circleMarker([lat, lng], {
                    radius: isLogger ? 8 : 6,
                    fillColor: isLogger ? loggerColor : teamColor,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: isLogger ? 0.9 : 0.5
                });
                marker.bindTooltip(holeId, { permanent: false, direction: 'top' });
                if (isLogger) loggerLayer.addLayer(marker);
                else teamLayer.addLayer(marker);
            });
            teamLayer.addTo(map);
            loggerLayer.addTo(map);

            if (bounds && bounds.length === 4) {
                map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: [20, 20], maxZoom: 15 });
            } else {
                const lats = points.map(p => p.lat);
                const lngs = points.map(p => p.lng);
                map.fitBounds([[Math.min.apply(null, lats), Math.min.apply(null, lngs)], [Math.max.apply(null, lats), Math.max.apply(null, lngs)]], { padding: [20, 20], maxZoom: 15 });
            }
        }

        setupLeafletMap();

        // Print functionality - show all tabs for PDF export
        function printReport() {
            const activeTabId = document.querySelector('.tab-button.active')?.dataset.tabButton || 'overview';
            // Show all tab panels for printing
            document.querySelectorAll('.tab-panel').forEach(panel => {
                panel.style.display = 'block';
            });
            // Initialize all Plotly charts (they may not have been rendered yet)
            initPlotlyCharts();
            // Small delay to ensure charts render
            setTimeout(() => {
                window.print();
                // Restore tab panel display after printing
                setTimeout(() => {
                    document.querySelectorAll('.tab-panel').forEach(panel => {
                        panel.style.display = '';
                    });
                    activateTab(activeTabId);
                }, 500);
            }, 300);
        }

        // Add print button click handler
        const printBtn = document.getElementById('print-report-btn');
        if (printBtn) {
            printBtn.addEventListener('click', printReport);
        }

        // Image expansion handler - ensures only one image is expanded at a time; highlights source row
        function clearRowHighlight() {
            document.querySelectorAll('tr.source-row-highlight').forEach(function(row) {
                row.classList.remove('source-row-highlight');
            });
        }
        function handleImageExpand(img) {
            const wasExpanded = img.classList.contains('expanded');
            // Clear any previous row highlight and close all expanded images first
            clearRowHighlight();
            document.querySelectorAll('.expandable-img.expanded').forEach(function(other) {
                other.classList.remove('expanded');
            });
            // Toggle the clicked image (if it wasn't already expanded, expand it)
            if (!wasExpanded) {
                img.classList.add('expanded');
                const row = img.closest('tr');
                if (row) row.classList.add('source-row-highlight');
            }
        }

        // Close expanded images when clicking outside
        document.addEventListener('click', function(e) {
            // If clicking on an expandable image, let the onclick handler manage it
            if (e.target.classList.contains('expandable-img')) {
                return;
            }
            // Otherwise, close all expanded images and clear row highlight
            clearRowHighlight();
            document.querySelectorAll('.expandable-img.expanded').forEach(function(img) {
                img.classList.remove('expanded');
            });
        });

        // Sortable table functionality
        function initSortableTables() {
            document.querySelectorAll('.sortable-table').forEach(function(table) {
                const headers = table.querySelectorAll('th.sortable');
                headers.forEach(function(header) {
                    header.addEventListener('click', function() {
                        const tbody = table.querySelector('tbody');
                        if (!tbody) return;
                        const colIndex = Number(header.dataset.colIndex || header.cellIndex);
                        const isAsc = header.classList.contains('sort-asc');

                        // Remove sort classes from all headers
                        headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));

                        const compareRows = function(a, b) {
                            const cellA = a.children[colIndex];
                            const cellB = b.children[colIndex];
                            let valA = cellA ? (cellA.dataset.sort || cellA.textContent.trim()) : '';
                            let valB = cellB ? (cellB.dataset.sort || cellB.textContent.trim()) : '';

                            // Try numeric comparison
                            const numA = parseFloat(valA);
                            const numB = parseFloat(valB);
                            if (!isNaN(numA) && !isNaN(numB)) {
                                return isAsc ? numB - numA : numA - numB;
                            }

                            // String comparison
                            return isAsc ? valB.localeCompare(valA) : valA.localeCompare(valB);
                        };

                        if (table.id === 'all-issues-table') {
                            const groups = [];
                            let currentGroup = null;
                            Array.from(tbody.querySelectorAll('tr')).forEach(function(row) {
                                if (row.classList.contains('all-issues-hole-header')) {
                                    currentGroup = { header: row, rows: [] };
                                    groups.push(currentGroup);
                                } else if (currentGroup) {
                                    currentGroup.rows.push(row);
                                } else {
                                    groups.push({ header: null, rows: [row] });
                                }
                            });
                            groups.forEach(function(group) {
                                group.rows.sort(compareRows);
                                if (group.header) tbody.appendChild(group.header);
                                group.rows.forEach(row => tbody.appendChild(row));
                            });
                        } else {
                            const rows = Array.from(tbody.querySelectorAll('tr:not(.all-issues-hole-header)'));
                            rows.sort(compareRows);
                            rows.forEach(row => tbody.appendChild(row));
                        }

                        header.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
                    });
                });
            });
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                initPlotlyCharts();
                resizePlotlyInActivePanel();
                initSortableTables();
            });
        } else {
            initPlotlyCharts();
            resizePlotlyInActivePanel();
            initSortableTables();
        }
"""
