// MacH3 Metal frontend helpers:
// - the Take's Mode and Continue switch the input groups on and off. Loaders a take
//   does not read are muted (greyed, never run) and their group goes grey; switching
//   back restores each node to the state it had, so a picture you bypassed stays so.
// - while h3c Settings is wired into a Take, the Take hides the fields Settings sets.
// - h3c Settings shows the size its megapixels and aspect give, and its lane switch
//   sets the steps to match.
// - the key dials (mode, prompt, length, seed; size, lane, continue) are drawn in
//   amber, bold and in capitals, so they stand out from the rest.
import { app } from "../../scripts/app.js";

// Mirrors MAIN_MODES / GENERATE_MODES in nodes.py.
const MODES = {
  "T2V · text to video": [],
  "I2V · first frame": ["first_frame"],
  "FL2V · first + last frame": ["first_frame", "last_frame"],
  "R2V · references": ["references"],
  "auto": ["first_frame", "last_frame", "references", "motion_context"],
  "text to video": [],
  "first frame": ["first_frame"],
  "first + last frame": ["first_frame", "last_frame"],
  "references": ["references"],
  "references + first frame": ["references", "first_frame"],
  "references + first + last frame": ["references", "first_frame", "last_frame"],
  "continue": ["motion_context"],
  "continue + references": ["motion_context", "references"],
  "continue + references + last frame": ["motion_context", "references", "last_frame"],
};
const INPUTS = ["first_frame", "last_frame", "references", "motion_context"];
const MUTED = 2;
const GREY = "#444444";

function settingsOf(take) {
  const slot = take.inputs?.findIndex((i) => i.name === "settings") ?? -1;
  return slot >= 0 ? take.getInputNode(slot) : null;
}

function wantedInputs(take) {
  const get = (node, name) => node?.widgets?.find((w) => w.name === name)?.value;
  const wanted = new Set(MODES[get(take, "mode")] ?? INPUTS);
  // Continue sits on h3c Settings when one is wired in (as in nodes.py).
  const cont = get(settingsOf(take), "continue_from") ?? get(take, "continue_from") ?? "off";
  if (cont !== "off") {
    wanted.delete("first_frame");
    wanted.delete("motion_context");
    if (cont === "clip input") wanted.add("motion_context");
  }
  return wanted;
}

function upstream(node, seen = new Set()) {
  for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
    const src = node.getInputNode?.(i);
    if (src && !seen.has(src.id)) {
      seen.add(src.id);
      upstream(src, seen);
    }
  }
  return seen;
}

function inside(group, node) {
  const [gx, gy, gw, gh] = group._bounding ?? group.bounding;
  const [x, y] = node.pos;
  return x >= gx && y >= gy && x < gx + gw && y < gy + gh;
}

function apply(take) {
  const graph = take.graph;
  if (!graph) return;
  const wanted = wantedInputs(take);
  const on = new Set(), off = new Set();
  for (const name of INPUTS) {
    const slot = take.inputs?.findIndex((i) => i.name === name) ?? -1;
    const src = slot >= 0 ? take.getInputNode(slot) : null;
    if (!src) continue;
    const ids = upstream(src, new Set([src.id]));
    for (const id of ids) (wanted.has(name) ? on : off).add(id);
  }
  for (const id of on) off.delete(id);  // a loader shared with a used input stays on
  // The Take records what it muted and the mode each node had, so everything comes
  // back on the next pass, even a node that was disconnected in between.
  take.properties ??= {};
  const before = take.properties.h3c_muted ?? {};
  for (const [id, mode] of Object.entries(before)) {
    const n = graph.getNodeById(Number(id)) ?? graph.getNodeById(id);
    if (n && n.mode === MUTED) n.mode = mode;
  }
  const muted = {};
  for (const id of off) {
    const n = graph.getNodeById(id);
    if (!n) continue;
    muted[id] = n.mode;
    n.mode = MUTED;
  }
  take.properties.h3c_muted = muted;
  // Groups: grey when nothing in them is in use. The text-to-video group holds
  // only a note, so it lights up when the take reads no pictures at all.
  const colors = (take.properties.h3c_group_colors ??= {});
  for (const group of graph._groups ?? []) {
    const nodes = graph._nodes.filter((n) => inside(group, n));
    let active;
    if (nodes.some((n) => on.has(n.id))) active = true;
    else if (nodes.some((n) => off.has(n.id))) active = false;
    else if (/^T2V\b/.test(group.title)) active = wanted.size === 0;
    else continue;
    const key = String(group.id ?? group.title);
    if (!(key in colors) && group.color !== GREY) colors[key] = group.color;
    group.color = active ? (colors[key] ?? group.color) : GREY;
  }
  graph.setDirtyCanvas?.(true, true);
}

// Mirrors H3cSettings.OVERRIDES and size_for() in nodes.py.
const OVERRIDES = ["width", "height", "steps", "continue_from"];
const ASPECTS = { "9:16 portrait": 9 / 16, "16:9 landscape": 16 / 9, "1:1 square": 1 };
// Steps a lane switch sets (as in the MacH3 Metal app): Turbo is tuned for 5-6.
const LANE_STEPS = { quality: 12, speed: 6 };

function sizeFor(megapixels, aspect) {
  const px = Number(megapixels) * 1024 * 1024, a = ASPECTS[aspect] ?? 9 / 16;
  const r32 = (v) => Math.max(64, Math.round(v / 32) * 32);
  let w = r32(Math.sqrt(px * a)), h = r32(Math.sqrt(px / a));
  while (w * h > 768 * 1344) {
    if (w > h) w -= 32; else if (h > w) h -= 32; else { w -= 32; h -= 32; }
  }
  return [w, h];
}

function hideOverridden(take, resize = true) {
  const wired = settingsOf(take) != null;
  let changed = false;
  for (const w of take.widgets ?? []) {
    if (!OVERRIDES.includes(w.name) || !!w.hidden === wired) continue;
    w.hidden = wired;
    changed = true;
  }
  // Never shrink (the prompt box keeps the freed space); grow when fields come back.
  if (changed && resize) take.setSize([take.size[0], Math.max(take.size[1], take.computeSize()[1])]);
}

// The key dials per node, and the label each is drawn with.
const KEYS = {
  H3cGenerate: { mode: "MODE", prompt: null, seconds: "SECONDS", seed: "SEED" },
  H3cSettings: { megapixels: "MEGAPIXELS", aspect: "ASPECT", lane: "LANE", continue_from: "CONTINUE" },
};
const KEY_COLORS = { background_color: "#3a2a10", outline_color: "#f0a830",
  secondary_text_color: "#f5c26b", text_color: "#ffffff" };

function bold(widget) {
  const draw = Object.getPrototypeOf(widget).drawWidget;
  if (!draw || widget.h3cBold) return;
  widget.h3cBold = true;
  const font = Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, "font");
  widget.drawWidget = function (ctx, options) {
    // The widget sets its own font while it draws; make every font it sets bold.
    const before = ctx.font;
    Object.defineProperty(ctx, "font", { configurable: true,
      get: () => font.get.call(ctx),
      set: (v) => font.set.call(ctx, /bold|[6-9]00/.test(v) ? v : `bold ${v}`) });
    ctx.font = before;
    try {
      return draw.call(this, ctx, options);
    } finally {
      delete ctx.font;
      ctx.font = before;
    }
  };
}

function flagKeys(node) {
  const keys = KEYS[node.comfyClass];
  if (!keys) return;
  for (const widget of node.widgets ?? []) {
    if (!(widget.name in keys)) continue;
    if (widget.element) {  // the prompt box is a DOM textarea
      widget.element.style.outline = `2px solid ${KEY_COLORS.outline_color}`;
      continue;
    }
    for (const [k, v] of Object.entries(KEY_COLORS))
      Object.defineProperty(widget, k, { configurable: true, get: () => v });
    if (keys[widget.name] && !(widget.name === "megapixels" && widget.label)) widget.label = keys[widget.name];
    bold(widget);
  }
}

function showSize(settings) {
  const get = (name) => settings.widgets?.find((w) => w.name === name);
  const mp = get("megapixels");
  if (!mp) return;
  const [w, h] = sizeFor(mp.value, get("aspect")?.value);
  mp.label = `MEGAPIXELS (${w} x ${h})`;
  settings.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "MacH3Metal.ModeGroups",
  nodeCreated(node) {
    if (node.comfyClass === "H3cSettings") {
      const lane = node.widgets?.find((w) => w.name === "lane");
      const steps = node.widgets?.find((w) => w.name === "steps");
      if (lane && steps) {
        const original = lane.callback;
        lane.callback = function (...args) {
          const result = original?.apply(this, args);
          steps.value = String(lane.value).startsWith("Speed") ? LANE_STEPS.speed : LANE_STEPS.quality;
          node.setDirtyCanvas?.(true, true);
          return result;
        };
      }
      const cont = node.widgets?.find((w) => w.name === "continue_from");
      if (cont) {
        const original = cont.callback;
        cont.callback = function (...args) {
          const result = original?.apply(this, args);
          for (const take of node.graph?._nodes ?? [])
            if (take.comfyClass === "H3cGenerate" && settingsOf(take) === node) apply(take);
          return result;
        };
      }
      for (const name of ["megapixels", "aspect"]) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget) continue;
        const original = widget.callback;
        widget.callback = function (...args) {
          const result = original?.apply(this, args);
          showSize(node);
          return result;
        };
      }
      flagKeys(node);
      return;
    }
    if (node.comfyClass !== "H3cGenerate") return;
    flagKeys(node);
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      const result = onConnectionsChange?.apply(this, args);
      setTimeout(() => {
        hideOverridden(node);
        apply(node);
      }, 0);
      return result;
    };
    for (const name of ["mode", "continue_from"]) {
      const widget = node.widgets?.find((w) => w.name === name);
      if (!widget) continue;
      const original = widget.callback;
      widget.callback = function (...args) {
        const result = original?.apply(this, args);
        apply(node);
        return result;
      };
    }
  },
  afterConfigureGraph() {
    for (const node of app.graph?._nodes ?? []) {
      if (node.comfyClass === "H3cGenerate") {
        hideOverridden(node, false);
        apply(node);
      }
      if (node.comfyClass === "H3cSettings") showSize(node);
      flagKeys(node);  // after configure: saved labels must not undo the key labels
    }
  },
});
