// ---- SCOUTIQO PATH PRELUDE (injected) ----
window.SC_DATA_URL   = "/static/data.json";
window.SC_EVENTS_URL = "/static/events.json";
window.SC_VIDEO_URL  = "/media/source.mp4";
const _fetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  try {
    if (typeof input === "string") {
      if (input.endsWith("data.json"))   input = window.SC_DATA_URL;
      if (input.endsWith("events.json")) input = window.SC_EVENTS_URL;
    }
  } catch(e){}
  return _fetch(input, init);
};
// ---- END PRELUDE ----
