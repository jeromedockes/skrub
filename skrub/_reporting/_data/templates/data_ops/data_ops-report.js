(function() {
    const drawer = document.getElementById('drawer');
    const nodeView = document.getElementById('node-view');
    const sourceView = document.getElementById('source-view');
    const nodeFrameContainer = document.getElementById('node-frame-container');
    const sourceFrameContainer = document.getElementById('source-frame-container');
    const highlightStyle = document.getElementById('current-node-style');

    // The iframe showing the selected node. It is recreated each time we open
    // a node: navigating an existing iframe would add entries to the browser's
    // history, whereas loading a fresh one does not.
    let nodeFrame = null;
    const totalNNodes = Number(document.body.dataset.totalNodes);

    function makeFrame(description) {
        const frame = document.createElement('iframe');
        frame.title = description;
        return frame;
    }

    function nodeIdFromUrl(url) {
        const pathname = new URL(url, window.location.href).pathname;
        const match = /(?:^|\/)node_(\d+)\.html$/.exec(pathname);
        return match ? Number(match[1]) : null;
    }

    function nodeUrl(nodeId) {
        return new URL(`node_${nodeId}.html`, window.location.href).href;
    }

    function setNodeHash(nodeId) {
        const url = new URL(window.location.href);
        const hash = nodeId === null ? '' : `node_${nodeId}`;
        if (url.hash === (hash ? `#${hash}` : '')) {
            return;
        }
        url.hash = hash;
        window.history.replaceState(null, '', url.href);
    }

    function nodeIdFromHash() {
        const match = /^#node_(\d+)$/.exec(window.location.hash);
        if (!match) {
            return null;
        }
        const nodeId = Number(match[1]);
        if (!Number.isInteger(nodeId) || nodeId < 0 || nodeId >= totalNNodes) {
            return null;
        }
        return nodeId;
    }

    function openNode(url) {
        // If a source file is being displayed, discard it and go back to the
        // node view.
        closeSource();

        nodeFrameContainer.replaceChildren();
        nodeFrame = makeFrame('Details of the selected node');
        nodeFrame.name = 'node-frame';
        nodeFrame.src = url;
        nodeFrameContainer.append(nodeFrame);

        nodeView.hidden = false;
        drawer.hidden = false;
        document.body.classList.add('drawer-open');
        setNodeHash(nodeIdFromUrl(url));
    }

    function closeDrawer() {
        nodeFrameContainer.replaceChildren();
        nodeFrame = null;
        closeSource();
        drawer.hidden = true;
        document.body.classList.remove('drawer-open');
        highlightStyle.textContent = '';
        setNodeHash(null);
    }

    function openSource(url) {
        sourceFrameContainer.replaceChildren();
        const frame = makeFrame('Source code');
        frame.src = url;
        sourceFrameContainer.append(frame);

        nodeView.hidden = true;
        sourceView.hidden = false;
        drawer.hidden = false;
        document.body.classList.add('drawer-open');
    }

    function closeSource() {
        sourceFrameContainer.replaceChildren();
        sourceView.hidden = true;
        if (!drawer.hidden) {
            nodeView.hidden = false;
        }
    }

    document.getElementById('close-drawer').addEventListener('click', closeDrawer);
    document.getElementById('close-source').addEventListener('click', closeSource);

    // Node pages are displayed in the iframe and tell us which node they show
    // (see node.html). They also ask us to open other report pages: this lets
    // us replace the iframe instead of navigating it. We cannot read the
    // iframe's location ourselves: for file:// URLs browsers treat it as
    // cross-origin.
    //
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it does not matter whether the graph is already
    // there: it may be rendered later by the browser (when graphviz is not
    // installed).
    window.addEventListener('message', (event) => {
        if (!nodeFrame || event.source !== nodeFrame.contentWindow) {
            return;
        }
        if (!event.data || typeof event.data !== 'object') {
            return;
        }
        if (event.data.type === 'skrub-report-node-shown') {
            const nodeId = event.data.nodeId;
            // nodeId is null for pages that do not show a particular node
            highlightStyle.textContent = (nodeId === null) ? '' : (
                `#node_${Number(nodeId)} polygon {` +
                ' fill: var(--current-node-color); stroke-width: 3; }');
        } else if (event.data.type === 'skrub-report-navigate') {
            openNode(event.data.url);
        } else if (event.data.type === 'skrub-report-source') {
            openSource(event.data.url);
        }
    });

    // Links in the graph target the iframe. Open the node drawer instead of
    // letting the browser navigate the iframe.
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
        openNode(new URL(href, window.location.href).href);
    });

    window.addEventListener('hashchange', () => {
        const nodeId = nodeIdFromHash();
        if (nodeId === null) {
            if (!drawer.hidden) {
                closeDrawer();
            }
        } else {
            openNode(nodeUrl(nodeId));
        }
    });

    const initialNodeId = nodeIdFromHash();
    if (initialNodeId !== null) {
        openNode(nodeUrl(initialNodeId));
    }

    const toggleNav = document.getElementById('toggle-nav');
    const toggleNavLabel = document.getElementById('toggle-nav-label');
    toggleNav.addEventListener('click', () => {
        const isOpen = document.querySelector('nav').toggleAttribute('data-is-open');
        toggleNav.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        toggleNav.title = isOpen ? 'Hide graph' : 'Show graph';
        toggleNavLabel.textContent = isOpen ? 'Hide graph' : 'Show graph';
    });
})();
