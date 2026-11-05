(function() {
    const drawer = document.getElementById('drawer');
    const nodeView = document.getElementById('node-view');
    const sourceView = document.getElementById('source-view');
    const nodeFrameContainer = document.getElementById('node-frame-container');
    const sourceFrameContainer = document.getElementById('source-frame-container');
    // Style element used to highlight the current node in the graph. We set a
    // CSS rule rather than an attribute on the node so that it works even if
    // the graph has not been rendered yet or the node cannot easily be found.
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
        return nodeId < totalNNodes ? nodeId : null;
    }

    function openNode(nodeId) {
        // If a source file is being displayed, discard it and go back to the
        // node view.
        closeSource();

        nodeFrameContainer.replaceChildren();
        nodeFrame = makeFrame('Details of the selected node');
        nodeFrame.name = 'node-frame';
        nodeFrame.src = nodeUrl(nodeId);
        nodeFrameContainer.append(nodeFrame);

        nodeView.hidden = false;
        drawer.hidden = false;
        document.body.classList.add('drawer-open');
        setNodeHash(nodeId);
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

    // Node pages tell us which node they show (see node.html) so that we can
    // update the graph highlight and the shareable URL. Source code links ask
    // us to open them in a separate drawer view. We cannot read the iframe's
    // location ourselves: for file:// URLs browsers treat it as cross-origin.
    //
    // The current node is highlighted with a CSS rule rather than by editing
    // the graph, so that it does not matter whether the graph is already
    // there: it may be rendered later by the browser (when graphviz is not
    // installed for the python process that creates the report, the graph is
    // drawn in the browser).
    window.addEventListener('message', (event) => {
        if (!nodeFrame || event.source !== nodeFrame.contentWindow) {
            return;
        }
        if (!event.data || typeof event.data !== 'object') {
            return;
        }
        if (event.data.type === 'skrub-report-node-shown') {
            const nodeId = Number(event.data.nodeId);
            highlightStyle.textContent = (
                `#node_${nodeId} polygon {` +
                ' fill: var(--current-node-color); stroke-width: 3; }');
            setNodeHash(nodeId);
        } else if (event.data.type === 'skrub-report-source') {
            openSource(event.data.url);
        }
    });

    // Graph nodes link to a hash such as "#node_3". Open the corresponding
    // node in the drawer ourselves so that following the link replaces the
    // current history entry rather than adding a new one.
    document.addEventListener('click', (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey ||
            event.altKey || !(event.target instanceof Element)) {
            return;
        }
        const link = event.target.closest('a');
        if (!link) {
            return;
        }
        const href = link.getAttribute('href') ||
            link.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
        const match = href && /^#node_(\d+)$/.exec(href);
        if (!match) {
            return;
        }
        event.preventDefault();
        openNode(Number(match[1]));
    });

    const initialNodeId = nodeIdFromHash();
    if (initialNodeId !== null) {
        openNode(initialNodeId);
    }

    // Button that shows or hides the graph panel.
    const toggleNav = document.getElementById('toggle-nav');
    // Label inside the graph toggle button.
    const toggleNavLabel = document.getElementById('toggle-nav-label');
    toggleNav.addEventListener('click', () => {
        const isOpen = document.querySelector('nav').toggleAttribute('data-is-open');
        toggleNav.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        toggleNav.title = isOpen ? 'Hide graph' : 'Show graph';
        toggleNavLabel.textContent = isOpen ? 'Hide graph' : 'Show graph';
    });
})();
