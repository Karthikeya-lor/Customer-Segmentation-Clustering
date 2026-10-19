/* Host integration only. The original website controls, data and inference are unchanged. */
(() => {
  if (window.parent === window) return;
  // Streamlit defaults component frames to scrolling="no". This is a complete page.
  try { if (window.frameElement) window.frameElement.setAttribute('scrolling', 'auto'); } catch (_) {}
  // Keep the original section links shareable on the Streamlit app URL as well.
  let host = window.parent;
  try { if (window.top.location.origin === window.location.origin) host = window.top; } catch (_) {}
  const sectionHash = value => /^(#(lab|tracks|workflow)?)?$/.test(value);
  try {
    if (host.location.hash && sectionHash(host.location.hash)) window.location.hash = host.location.hash;
    window.addEventListener('hashchange', () => {
      if (sectionHash(window.location.hash)) host.history.replaceState(null, '', host.location.pathname + host.location.search + window.location.hash);
    });
    host.addEventListener('hashchange', () => {
      if (sectionHash(host.location.hash) && host.location.hash !== window.location.hash) window.location.hash = host.location.hash;
    });
  } catch (_) { /* The original in-frame anchors still work on cross-origin hosts. */ }
  const send = (type, extra = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, '*');
  const resize = () => {
    let height = window.innerHeight;
    try { height = window.parent.innerHeight; } catch (_) { /* Cross-origin hosts retain iframe height. */ }
    send('streamlit:setFrameHeight', {height});
  };
  window.addEventListener('message', event => {
    if (event.source === window.parent && event.data?.type === 'streamlit:render') resize();
  });
  window.addEventListener('resize', resize);
  send('streamlit:componentReady', {apiVersion: 1});
  resize();
})();
