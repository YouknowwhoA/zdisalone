const state = {
  agents: [],
  selectedId: null,
  tasks: [],
  log: [],
  stats: {
    tokensIn: 0,
    tokensOut: 0,
    lastTokens: 0,
    totalCost: 0,
    priceIn: 0,
    priceOut: 0,
  },
};

const rosterEl = document.getElementById("roster");
const mapEl = document.getElementById("map-blips");
const selectedEl = document.getElementById("selected");
const skillTreeEl = document.getElementById("skill-tree");
const skillPointsEl = document.getElementById("skill-points");
const skillBranchEl = document.getElementById("skill-branch");
const logEl = document.getElementById("log");
const taskInput = document.getElementById("task-input");
const queueCountEl = document.getElementById("queue-count");
const onlineCountEl = document.getElementById("online-count");
const tokensInEl = document.getElementById("tokens-in");
const tokensOutEl = document.getElementById("tokens-out");
const tokensLastEl = document.getElementById("tokens-last");
const costTotalEl = document.getElementById("cost-total");
const priceInEl = document.getElementById("price-in");
const priceOutEl = document.getElementById("price-out");

const DEFAULT_SKILL_POINTS = 8;
const SKILL_TREES = [
  {
    branch: "Engineering",
    tiers: [
      [{ id: "eng-1", name: "Code Sprint", max: 3, desc: "Ship features faster." }],
      [
        { id: "eng-2", name: "Refactor Master", max: 3, prereq: ["eng-1"], desc: "Reduce tech debt." },
        { id: "eng-3", name: "Toolsmith", max: 2, prereq: ["eng-1"], desc: "Build automation tools." },
      ],
      [
        {
          id: "eng-4",
          name: "System Architect",
          max: 2,
          prereq: ["eng-2", "eng-3"],
          desc: "Design large systems.",
        },
        { id: "eng-5", name: "Deployment Ace", max: 2, prereq: ["eng-3"], desc: "Ship reliably." },
      ],
    ],
  },
  {
    branch: "Strategy",
    tiers: [
      [{ id: "str-1", name: "Problem Framing", max: 3, desc: "Define the real goal." }],
      [
        { id: "str-2", name: "Task Decomposition", max: 3, prereq: ["str-1"], desc: "Break down missions." },
        { id: "str-3", name: "Risk Scout", max: 2, prereq: ["str-1"], desc: "Spot weak links." },
      ],
      [{ id: "str-4", name: "Roadmap Vision", max: 2, prereq: ["str-2"], desc: "Long-term planning." }],
    ],
  },
  {
    branch: "Ops",
    tiers: [
      [{ id: "ops-1", name: "Automation Loops", max: 3, desc: "Reduce manual work." }],
      [
        { id: "ops-2", name: "Monitoring Sense", max: 3, prereq: ["ops-1"], desc: "Spot issues fast." },
        { id: "ops-3", name: "Incident Triage", max: 2, prereq: ["ops-1"], desc: "Stabilize quickly." },
      ],
      [{ id: "ops-4", name: "Reliability Playbook", max: 2, prereq: ["ops-2"], desc: "Run smoothly." }],
    ],
  },
  {
    branch: "Research",
    tiers: [
      [{ id: "res-1", name: "Signal Hunting", max: 3, desc: "Find strong sources." }],
      [
        { id: "res-2", name: "Source Verification", max: 3, prereq: ["res-1"], desc: "Validate truth." },
        { id: "res-3", name: "Synthesis", max: 2, prereq: ["res-1"], desc: "Summarize insights." },
      ],
      [{ id: "res-4", name: "Insight Report", max: 2, prereq: ["res-2", "res-3"], desc: "Deliver impact." }],
    ],
  },
  {
    branch: "Creative",
    tiers: [
      [{ id: "cre-1", name: "Visual Direction", max: 3, desc: "Set the vibe." }],
      [
        { id: "cre-2", name: "UI Flow", max: 3, prereq: ["cre-1"], desc: "Smooth interactions." },
        { id: "cre-3", name: "Motion Craft", max: 2, prereq: ["cre-1"], desc: "Add cinematic motion." },
      ],
      [{ id: "cre-4", name: "Brand Aura", max: 2, prereq: ["cre-2"], desc: "Memorable identity." }],
    ],
  },
];

const NODE_INDEX = (() => {
  const index = {};
  SKILL_TREES.forEach((branch) => {
    branch.tiers.forEach((row) => {
      row.forEach((node) => {
        index[node.id] = { ...node, branch: branch.branch };
      });
    });
  });
  return index;
})();

function addLog(message) {
  const timestamp = new Date().toLocaleTimeString();
  state.log.unshift({ message, timestamp });
  if (state.log.length > 12) {
    state.log.pop();
  }
  renderLog();
}

function normalizeAgent(agent) {
  const existing = state.agents.find((a) => a.id === agent.id);
  const location = existing?.location || {
    x: 10 + Math.random() * 80,
    y: 15 + Math.random() * 70,
  };
  const stored = loadAgentProgress(agent.id);
  const skills = Array.isArray(agent.skills) ? agent.skills : [];
  const skillPoints = agent.skillPoints ?? stored?.points ?? DEFAULT_SKILL_POINTS;
  const skillTree = agent.skillTree ?? stored?.tree ?? {};
  return {
    ...agent,
    location,
    skills,
    skillPoints,
    skillTree,
    lastSeen: agent.lastSeen || "just now",
    mock: false,
    emoji: agent.emoji || null,
    color: agent.color || null,
    tagline: agent.tagline || null,
    agentType: agent.agentType || "api",
    agentTypeLabel: agent.agentTypeLabel || null,
  };
}

function loadAgentProgress(id) {
  try {
    const raw = localStorage.getItem(`agentProgress:${id}`);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (err) {
    return null;
  }
}

function saveAgentProgress(agent) {
  try {
    localStorage.setItem(
      `agentProgress:${agent.id}`,
      JSON.stringify({ points: agent.skillPoints, tree: agent.skillTree })
    );
  } catch (err) {
    // ignore storage issues
  }
}

async function syncAgents() {
  try {
    const res = await fetch("/api/agents");
    if (!res.ok) {
      throw new Error(`Agent fetch failed (${res.status})`);
    }
    const payload = await res.json();
    let agents = (payload.agents || []).map(normalizeAgent);
    if (!agents.length) {
      addLog("No agents returned from server.");
      return;
    }
    // 保底：如果服务器还没返回「小光标」，前端先加一个，这样刷新就能看到
    if (!agents.some((a) => a.id === "cursor-auto")) {
      agents = agents.concat([
        normalizeAgent({
          id: "cursor-auto",
          name: "Auto · 小光标",
          status: "idle",
          skills: ["code", "edit", "search", "chat", "卖萌"],
          lastSeen: "just now",
          emoji: "✨",
          color: "cursor",
          tagline: "会写代码会卖萌",
          agentType: "display",
          agentTypeLabel: "In-IDE · 展示",
        }),
      ]);
      addLog("Auto · 小光标 joined the squad (frontend fallback).");
    }
    state.agents = agents;
    if (!state.selectedId || !state.agents.find((a) => a.id === state.selectedId)) {
      state.selectedId = state.agents[0].id;
    }
    addLog("Synced agents from adapter.");
    renderAll();
  } catch (err) {
    addLog(`Adapter offline: ${err.message}`);
  }
}

function renderRoster() {
  rosterEl.innerHTML = "";
  state.agents.forEach((agent) => {
    const card = document.createElement("div");
    card.className = "roster-card" + (agent.id === state.selectedId ? " active" : "");
    card.addEventListener("click", () => selectAgent(agent.id));

    if (agent.color) {
      card.classList.add(agent.color);
    }
    const header = document.createElement("div");
    header.className = "roster-card-header";

    const name = document.createElement("div");
    name.className = "roster-name";
    if (agent.emoji) {
      const emojiSpan = document.createElement("span");
      emojiSpan.className = "roster-emoji";
      emojiSpan.textContent = agent.emoji + " ";
      name.appendChild(emojiSpan);
    }
    name.appendChild(document.createTextNode(agent.name));

    header.appendChild(name);

    if (agent.mock) {
      const remove = document.createElement("button");
      remove.className = "roster-remove";
      remove.textContent = "Remove";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        removeAgent(agent.id);
      });
      header.appendChild(remove);
    }

    const meta = document.createElement("div");
    meta.className = "roster-meta";
    const metaText = agent.tagline
      ? `${agent.tagline} · ${agent.status} · ${agent.lastSeen}`
      : `${agent.status.toUpperCase()} · last seen ${agent.lastSeen}`;
    meta.appendChild(document.createTextNode(metaText));
    if (agent.agentTypeLabel) {
      meta.appendChild(document.createElement("br"));
      const badge = document.createElement("span");
      badge.className = "roster-badge " + (agent.agentType || "api");
      badge.textContent = agent.agentTypeLabel;
      meta.appendChild(badge);
    }

    card.appendChild(header);
    card.appendChild(meta);
    rosterEl.appendChild(card);
  });
}

function renderMap() {
  mapEl.innerHTML = "";
  state.agents.forEach((agent) => {
    const blip = document.createElement("div");
    blip.className = `blip ${agent.status}` + (agent.color ? ` ${agent.color}` : "");
    if (agent.emoji) {
      const label = document.createElement("span");
      label.className = "blip-emoji";
      label.textContent = agent.emoji;
      blip.appendChild(label);
    }
    blip.style.left = `${agent.location.x}%`;
    blip.style.top = `${agent.location.y}%`;
    blip.title = agent.tagline ? `${agent.name} · ${agent.tagline}` : `${agent.name} (${agent.status})`;
    blip.addEventListener("click", () => selectAgent(agent.id));
    mapEl.appendChild(blip);
  });
}

function renderSelected() {
  const agent = state.agents.find((a) => a.id === state.selectedId);
  if (!agent) {
    selectedEl.innerHTML = `
      <div class="selected-title">No agent selected</div>
      <div class="selected-sub">Select an agent to dispatch a task.</div>
    `;
    selectedEl.classList.remove("cursor");
    skillTreeEl.innerHTML = "";
    skillPointsEl.textContent = "0";
    skillBranchEl.textContent = "-";
    return;
  }

  const subText = agent.tagline
    ? `${agent.tagline} · ${agent.status} · ${agent.lastSeen}`
    : `Status: ${agent.status.toUpperCase()} · last seen ${agent.lastSeen}`;
  const typeHtml = agent.agentTypeLabel
    ? `<div class="selected-type"><span class="selected-badge ${agent.agentType || "api"}">${agent.agentTypeLabel}</span></div>`
    : "";
  selectedEl.innerHTML = `
    <div class="selected-title">${agent.emoji ? agent.emoji + " " : ""}${agent.name}</div>
    <div class="selected-sub">${subText}</div>
    ${typeHtml}
  `;
  if (agent.color) {
    selectedEl.classList.add(agent.color);
  } else {
    selectedEl.classList.remove("cursor");
  }

  renderSkillTree(agent);
}

function renderSkillTree(agent) {
  skillTreeEl.innerHTML = "";
  const progress = agent.skillTree || {};
  if (!SKILL_TREES.length) {
    skillTreeEl.textContent = "No skills tracked.";
    skillPointsEl.textContent = "0";
    skillBranchEl.textContent = "-";
    return;
  }

  const spent = Object.values(progress).reduce((sum, value) => sum + value, 0);
  const available = Math.max(0, agent.skillPoints - spent);
  skillPointsEl.textContent = `${available} / ${agent.skillPoints}`;

  const branchTotals = {};
  SKILL_TREES.forEach((branch) => {
    branchTotals[branch.branch] = 0;
    branch.tiers.forEach((row) => {
      row.forEach((node) => {
        branchTotals[branch.branch] += progress[node.id] || 0;
      });
    });
  });
  const primaryEntry = Object.entries(branchTotals).sort((a, b) => b[1] - a[1])[0];
  const primaryBranch = primaryEntry && primaryEntry[1] > 0 ? primaryEntry[0] : "-";
  skillBranchEl.textContent = primaryBranch;

  SKILL_TREES.forEach((branch) => {
    const branchEl = document.createElement("div");
    branchEl.className = "skill-branch";
    if (primaryBranch !== "-" && branch.branch === primaryBranch) {
      branchEl.classList.add("highlight");
    }

    const title = document.createElement("div");
    title.className = "branch-title";
    title.textContent = branch.branch;
    branchEl.appendChild(title);

    const grid = document.createElement("div");
    grid.className = "branch-grid";

    branch.tiers.forEach((row, rowIndex) => {
      const rowEl = document.createElement("div");
      rowEl.className = "branch-row";

      row.forEach((node) => {
        const level = progress[node.id] || 0;
        const unlocked = isNodeUnlocked(node, progress);
        const canUpgrade = unlocked && level < node.max && available > 0;
        const nodeEl = document.createElement("div");
        nodeEl.className = "skill-node";
        if (!unlocked) nodeEl.classList.add("locked");
        if (canUpgrade) nodeEl.classList.add("upgradeable");
        if (level >= node.max) nodeEl.classList.add("maxed");

        if (rowIndex > 0) {
          const connector = document.createElement("div");
          connector.className = "node-connector";
          nodeEl.appendChild(connector);
        }

        const titleEl = document.createElement("div");
        titleEl.className = "node-title";
        titleEl.textContent = node.name;

        const bar = document.createElement("div");
        bar.className = "node-bar";
        const fill = document.createElement("span");
        fill.style.width = `${(level / node.max) * 100}%`;
        bar.appendChild(fill);

        const levelLabel = document.createElement("div");
        levelLabel.className = "node-level";
        levelLabel.textContent = `Lv ${level}/${node.max}`;

        const desc = document.createElement("div");
        desc.className = "node-desc";
        desc.textContent = node.desc;

        nodeEl.appendChild(titleEl);
        nodeEl.appendChild(bar);
        nodeEl.appendChild(levelLabel);
        nodeEl.appendChild(desc);

        if (!unlocked && node.prereq?.length) {
          const reqNames = node.prereq.map((id) => NODE_INDEX[id]?.name || id).join(", ");
          nodeEl.title = `Requires: ${reqNames}`;
        }

        nodeEl.addEventListener("click", () => {
          if (!canUpgrade) {
            return;
          }
          const next = level + 1;
          agent.skillTree = { ...progress, [node.id]: next };
          saveAgentProgress(agent);
          addLog(`Upgraded ${agent.name}: ${node.name} Lv ${next}/${node.max}`);
          renderAll();
        });

        rowEl.appendChild(nodeEl);
      });

      grid.appendChild(rowEl);
    });

    branchEl.appendChild(grid);
    skillTreeEl.appendChild(branchEl);
  });
}

function isNodeUnlocked(node, progress) {
  if (!node.prereq || node.prereq.length === 0) return true;
  return node.prereq.every((id) => (progress[id] || 0) > 0);
}

function renderLog() {
  logEl.innerHTML = "";
  state.log.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "log-item";
    item.textContent = `[${entry.timestamp}] ${entry.message}`;
    logEl.appendChild(item);
  });
}

function renderStats() {
  onlineCountEl.textContent = String(state.agents.length);
  queueCountEl.textContent = String(state.tasks.filter((t) => t.status === "queued").length);
  tokensInEl.textContent = String(state.stats.tokensIn);
  tokensOutEl.textContent = String(state.stats.tokensOut);
  tokensLastEl.textContent = String(state.stats.lastTokens);
  costTotalEl.textContent = `$${state.stats.totalCost.toFixed(4)}`;
}

function renderAll() {
  renderRoster();
  renderMap();
  renderSelected();
  renderStats();
}

function selectAgent(id) {
  state.selectedId = id;
  renderAll();
}

function removeAgent(id) {
  state.agents = state.agents.filter((agent) => agent.id !== id);
  if (state.selectedId === id) {
    state.selectedId = state.agents[0]?.id || null;
  }
  addLog(`Removed agent ${id}.`);
  renderAll();
}

async function dispatchTask() {
  const agent = state.agents.find((a) => a.id === state.selectedId);
  const text = taskInput.value.trim();
  if (!agent) {
    addLog("No agent selected. Task not dispatched.");
    return;
  }
  if (!text) {
    addLog("Task is empty.");
    return;
  }

  const task = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
    agentId: agent.id,
    text,
    status: "queued",
  };
  state.tasks.push(task);
  taskInput.value = "";
  addLog(`Queued task for ${agent.name}: ${text}`);
  renderStats();

  try {
    task.status = "running";
    agent.status = "running";
    renderAll();

    const res = await fetch("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agent.id, task: task.text }),
    });

    if (!res.ok) {
      const errPayload = await res.json().catch(() => ({}));
      throw new Error(errPayload.error || `Request failed (${res.status})`);
    }

    const payload = await res.json();
    task.status = payload.status || "done";
    agent.status = "idle";
    agent.lastSeen = "just now";

    if (payload.usage) {
      const promptTokens =
        payload.usage.prompt_tokens ??
        payload.usage.input_tokens ??
        payload.usage.input ??
        0;
      const completionTokens =
        payload.usage.completion_tokens ??
        payload.usage.output_tokens ??
        payload.usage.output ??
        0;
      const totalTokens =
        payload.usage.total_tokens ?? promptTokens + completionTokens;
      state.stats.tokensIn += promptTokens;
      state.stats.tokensOut += completionTokens;
      state.stats.lastTokens = totalTokens;
      updateCost();
    }

    addLog(`Task finished on ${agent.name} in ${payload.duration || "?"}s`);
    if (payload.output) {
      addLog(`Output: ${payload.output}`);
    }
  } catch (err) {
    task.status = "error";
    agent.status = "error";
    addLog(`Task failed on ${agent.name}: ${err.message}`);
  }

  renderAll();
}

function addMockAgent() {
  const id = `agent-${Math.random().toString(16).slice(2, 6)}`;
  const newAgent = {
    id,
    name: `unit-${state.agents.length + 1}`,
    status: "idle",
    skills: ["analysis", "automation", "scrape", "code", "ops"],
    skillPoints: DEFAULT_SKILL_POINTS,
    skillTree: {},
    location: { x: 10 + Math.random() * 80, y: 15 + Math.random() * 70 },
    lastSeen: "just now",
    mock: true,
    emoji: null,
    color: null,
    tagline: null,
  };
  state.agents.push(newAgent);
  addLog(`New agent online: ${newAgent.name}`);
  renderAll();
}

function clearMockAgents() {
  const before = state.agents.length;
  state.agents = state.agents.filter((agent) => !agent.mock);
  if (!state.agents.find((agent) => agent.id === state.selectedId)) {
    state.selectedId = state.agents[0]?.id || null;
  }
  const removed = before - state.agents.length;
  addLog(`Cleared ${removed} mock agents.`);
  renderAll();
}

function resetSkillTree() {
  const agent = state.agents.find((a) => a.id === state.selectedId);
  if (!agent) {
    addLog("No agent selected. Tree not reset.");
    return;
  }
  agent.skillTree = {};
  saveAgentProgress(agent);
  addLog(`Reset skill tree for ${agent.name}.`);
  renderAll();
}

function updateCost() {
  const priceIn = Number(state.stats.priceIn) || 0;
  const priceOut = Number(state.stats.priceOut) || 0;
  const inCost = (state.stats.tokensIn / 1000) * priceIn;
  const outCost = (state.stats.tokensOut / 1000) * priceOut;
  state.stats.totalCost = inCost + outCost;
  renderStats();
  localStorage.setItem("priceIn", String(state.stats.priceIn));
  localStorage.setItem("priceOut", String(state.stats.priceOut));
}

function boot() {
  addLog("Command console online.");
  state.stats.priceIn = Number(localStorage.getItem("priceIn")) || 0;
  state.stats.priceOut = Number(localStorage.getItem("priceOut")) || 0;
  priceInEl.value = state.stats.priceIn || "";
  priceOutEl.value = state.stats.priceOut || "";
  renderAll();
  syncAgents();
}

document.getElementById("dispatch").addEventListener("click", dispatchTask);
document.getElementById("simulate").addEventListener("click", addMockAgent);
document.getElementById("add-agent").addEventListener("click", addMockAgent);
document.getElementById("clear-mocks").addEventListener("click", clearMockAgents);
document.getElementById("reset-tree").addEventListener("click", resetSkillTree);

priceInEl.addEventListener("input", (event) => {
  state.stats.priceIn = Number(event.target.value) || 0;
  updateCost();
});

priceOutEl.addEventListener("input", (event) => {
  state.stats.priceOut = Number(event.target.value) || 0;
  updateCost();
});

document.getElementById("connect").addEventListener("click", () => {
  syncAgents();
});

boot();
