// MacH3 Metal: the Take's Mode and Continue switch the input groups on and off.
// Loaders a take does not read are muted (greyed, never run) and their group goes
// grey; switching back restores each node to the state it had, so a picture you
// bypassed yourself stays bypassed.
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

function wantedInputs(take) {
  const get = (name) => take.widgets?.find((w) => w.name === name)?.value;
  const wanted = new Set(MODES[get("mode")] ?? INPUTS);
  const cont = get("continue_from") ?? "off";
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
  for (const id of off) {
    const n = graph.getNodeById(id);
    if (!n) continue;
    n.properties ??= {};
    if (!n.properties.h3c_off) {
      n.properties.h3c_user_mode = n.mode;
      n.properties.h3c_off = true;
    }
    n.mode = MUTED;
  }
  for (const id of on) {
    const n = graph.getNodeById(id);
    if (!n?.properties?.h3c_off) continue;
    n.mode = n.properties.h3c_user_mode ?? 0;
    delete n.properties.h3c_off;
    delete n.properties.h3c_user_mode;
  }
  // Groups: grey when nothing in them is in use. The text-to-video group holds
  // only a note, so it lights up when the take reads no pictures at all.
  take.properties ??= {};
  const colors = (take.properties.h3c_group_colors ??= {});
  for (const group of graph._groups ?? []) {
    const nodes = graph._nodes.filter((n) => inside(group, n));
    let active;
    if (nodes.some((n) => on.has(n.id))) active = true;
    else if (nodes.some((n) => off.has(n.id))) active = false;
    else if (/^T2V\b/.test(group.title)) active = wanted.size === 0;
    else continue;
    if (!(group.title in colors)) colors[group.title] = group.color;
    group.color = active ? colors[group.title] : GREY;
  }
  graph.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "MacH3Metal.ModeGroups",
  nodeCreated(node) {
    if (node.comfyClass !== "H3cGenerate") return;
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
      if (node.comfyClass === "H3cGenerate") apply(node);
    }
  },
});
