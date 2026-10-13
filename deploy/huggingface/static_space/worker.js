'use strict';
/* Runs Mission control's Python in the visitor's own browser, in a web worker: Pyodide (CPython compiled to WebAssembly) with the project's source,
   experiments and stored energies unpacked from bundle.zip. It answers the page's /api requests with the same code the local server uses (substrate.gui.api),
   one at a time, and a job runs to its end inside the request that started it, because a browser worker has no threads to run it on.
   bundle.zip is fetched by the page and handed over, not fetched here: on a private Space the host authorises the page's own requests but refuses a worker's (a 401, seen on Hugging Face). */
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
importScripts(PYODIDE + 'pyodide.js');

const say = text => postMessage({ type: 'status', text });

const BOOT = `
import os, sys
sys.path.insert(0, "/bundle/src")
os.environ["SUBSTRATE_CACHE_DIR"] = "/bundle/cache"
os.environ["SUBSTRATE_MAX_ENSEMBLE"] = "200"
from substrate.gui import api
from substrate.gui.app import App
_app = App("/bundle", inline_jobs=True)

def handle(method, target, body):
    status, text = api.answer(_app, method, target, body)
    return f"{status}\\n{text}"
`;

let handle = null;
let failure = null;
let giveBundle;
const bundle = new Promise(resolve => { giveBundle = resolve; });

const ready = (async () => {
  say('loading Python (the first visit downloads about 40 MB)…');
  const py = await loadPyodide({ indexURL: PYODIDE });
  say('loading numpy and scipy…');
  await py.loadPackage(['numpy', 'scipy', 'pyyaml', 'sqlite3']);
  say('waiting for the models and the stored energies…');
  const bytes = await bundle;
  say('unpacking the models and the stored energies…');
  py.unpackArchive(bytes, 'zip', { extractDir: '/bundle' });
  say('starting…');
  py.runPython(BOOT);
  handle = py.globals.get('handle');
  say('ready');
})().catch(error => {
  failure = `Python could not be started in this browser: ${error && error.message ? error.message : error}`;
  say(failure);
});

onmessage = async ({ data }) => {
  if (data.type === 'bundle') return giveBundle(data.bytes);
  const { id, method, target, body } = data;
  const reply = (status, text) => postMessage({ id, status, text });
  await ready;
  if (failure) return reply(503, JSON.stringify({ error: failure }));
  try {
    const out = handle(method, target, body === undefined ? null : body);
    const split = out.indexOf('\n');
    reply(Number(out.slice(0, split)), out.slice(split + 1));
  } catch (error) {
    reply(500, JSON.stringify({ error: String(error && error.message ? error.message : error) }));
  }
};
