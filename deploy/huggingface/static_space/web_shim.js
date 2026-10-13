'use strict';
/* In the static demo there is no server: the page's calls to /api/... are handed to a web worker that runs the project's Python (worker.js) in this browser.
   This must load before app.js, which only ever calls fetch('/api/...'). Everything else (the page's own files) still goes to the network as usual. */
(() => {
  const realFetch = window.fetch.bind(window);
  const pending = new Map();
  let worker = null;
  let failure = null;
  let counter = 0;

  const say = text => { const pill = document.getElementById('status'); if (pill) pill.textContent = text; };
  const response = (status, text) => new Response(text, { status, headers: { 'Content-Type': 'application/json' } });

  function fail(message) {
    failure = message;
    say('Python could not start');
    for (const resolve of pending.values()) resolve(response(503, JSON.stringify({ error: message })));
    pending.clear();
  }

  function start() {
    if (worker || failure) return;
    try {
      worker = new Worker('worker.js');
    } catch (error) {
      return fail(`this browser could not start the Python worker: ${error.message}`);
    }
    worker.onmessage = ({ data }) => {
      if (data.type === 'status') return say(data.text);
      const resolve = pending.get(data.id);
      if (resolve) { pending.delete(data.id); resolve(response(data.status, data.text)); }
    };
    worker.onerror = event => fail(`the Python worker stopped: ${event.message || 'it could not be loaded'}`);
  }

  function ask(method, target, body) {
    start();
    if (failure) return Promise.resolve(response(503, JSON.stringify({ error: failure })));
    return new Promise(resolve => {
      const id = ++counter;
      pending.set(id, resolve);
      worker.postMessage({ id, method, target, body: typeof body === 'string' ? body : undefined });
    });
  }

  window.fetch = (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin === location.origin && url.pathname.startsWith('/api/')) return ask((init && init.method) || 'GET', url.pathname + url.search, init && init.body);
    return realFetch(input, init);
  };
})();
