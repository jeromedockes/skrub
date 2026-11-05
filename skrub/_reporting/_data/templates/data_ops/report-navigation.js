// Node pages are displayed in the drawer iframe of index.html. Follow links
// with location.replace() so that navigating between pages does not add
// entries to the browser's back-button history. Source code pages are opened
// in a separate drawer view (by the parent page) so that they can be
// dismissed to return to the node details.
(function () {
    const here = new URL(window.location.href);
    const reportDir = here.pathname.slice(
        0, here.pathname.lastIndexOf("/") + 1);
    const sourceDir = new URL("python/", here).pathname;
    function classify(url) {
        const target = new URL(url, here);
        if (target.protocol !== here.protocol ||
            target.host !== here.host ||
            !target.pathname.startsWith(reportDir)) {
            return null;
        }
        return target.pathname.startsWith(sourceDir) ? "source" : "node";
    }
    function follow(url) {
        const kind = classify(url);
        if (kind === null) {
            return false;
        }
        url = new URL(url, here).href;
        if (kind === "source" && window.parent !== window) {
            window.parent.postMessage({
                type: "skrub-report-source",
                url: url,
            }, "*");
        } else {
            window.location.replace(url);
        }
        return true;
    }
    document.addEventListener("click", (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey ||
            event.altKey || !(event.target instanceof Element)) {
            return;
        }
        const link = event.target.closest("a");
        if (!link) {
            return;
        }
        const target = link.getAttribute("target");
        const href = link.getAttribute("href") ||
            link.getAttributeNS("http://www.w3.org/1999/xlink", "href");
        if ((target && target !== "_self") || !href ||
            link.hasAttribute("download")) {
            return;
        }
        if (follow(href)) {
            event.preventDefault();
        }
    });
})();
