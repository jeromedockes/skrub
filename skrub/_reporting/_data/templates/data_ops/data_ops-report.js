(function() {
    // Node pages are displayed in the iframe and tell us which node they show
    // (see node.html). We cannot read the iframe's location ourselves: for
    // file:// URLs browsers treat it as cross-origin.
    //
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it does not matter whether the graph is already
    // there: it may be rendered later by the browser (when graphviz is not
    // installed).
    const frame = document.getElementById('node-frame');
    const highlightStyle = document.getElementById('current-node-style');

    window.addEventListener('message', (event) => {
        if (event.source !== frame.contentWindow) {
            return;
        }
        if (!event.data || event.data.type !== 'skrub-report-node-shown') {
            return;
        }
        const nodeId = event.data.nodeId;
        // nodeId is null for pages that do not show a particular node
        highlightStyle.textContent = (nodeId === null) ? '' : (
            `#node_${Number(nodeId)} polygon {` +
            ' fill: var(--current-node-color); stroke-width: 3; }');
    });

    // Links in the graph target the iframe. Instead of letting the browser
    // navigate it (which would add an entry to the history), ask the page in
    // the iframe to replace itself. Cross-origin restrictions (file:// pages
    // are treated as different origins) prevent us from changing the iframe's
    // location directly, hence the message.
    document.addEventListener('click', (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey ||
            event.altKey || !(event.target instanceof Element)) {
            return;
        }
        const link = event.target.closest('a');
        if (!link || link.getAttribute('target') !== 'node-frame') {
            return;
        }
        const href = link.getAttribute('href') ||
            link.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
        if (!href) {
            return;
        }
        event.preventDefault();
        frame.contentWindow.postMessage({
            type: 'skrub-report-navigate',
            url: new URL(href, window.location.href).href,
        }, '*');
    });

    document.getElementById('toggle-nav').addEventListener('click', () => {
        document.querySelector('nav').toggleAttribute('data-is-open');
    });
})();
