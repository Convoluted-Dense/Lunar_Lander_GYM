/* ==========================================================================
   REINFORCEMENT LEARNING PRESENTATION DECK — JAVASCRIPT CONTROLLER
   Features: Slide Navigation, Directional Animations, Presenter Notes, 
             Overview Grid, Audio Synthesis, GridWorld Simulator & Bandit
   ========================================================================== */

(function () {
  'use strict';

  // --- State Management ---
  const state = {
    currentSlide: 0,
    totalSlides: 7,
    isDarkTheme: true,
    isAutoplay: false,
    autoplayTimer: null,
    isSoundEnabled: true,
    audioCtx: null,
    gridworld: {
      agentPos: { r: 0, c: 0 },
      goalPos: { r: 3, c: 3 },
      hazards: [{ r: 1, c: 1 }, { r: 2, c: 2 }],
      totalReturn: 0,
      stepCount: 0,
      isAutoRunning: false,
      autoInterval: null,
      optimalPath: [
        { r: 0, c: 0, a: 'START', rwd: 0 },
        { r: 0, c: 1, a: 'RIGHT &rarr;', rwd: -0.1 },
        { r: 0, c: 2, a: 'RIGHT &rarr;', rwd: -0.1 },
        { r: 0, c: 3, a: 'RIGHT &rarr;', rwd: -0.1 },
        { r: 1, c: 3, a: 'DOWN &darr;', rwd: -0.1 },
        { r: 2, c: 3, a: 'DOWN &darr;', rwd: -0.1 },
        { r: 3, c: 3, a: 'DOWN &darr; (GOAL)', rwd: 10.0 }
      ],
      pathIndex: 0
    },
    bandit: {
      epsilon: 0.20,
      trueRates: [0.35, 0.78, 0.52, 0.22],
      pullCounts: [0, 0, 0, 0],
      totalRewards: [0, 0, 0, 0]
    }
  };

  // --- Presenter Notes Library ---
  const presenterNotes = [
    {
      title: "Title & Overview",
      notes: "Introduce Reinforcement Learning (RL) as learning from trial and error. Contrast it with Supervised learning: RL doesn't need labeled answers; it learns by taking actions and getting rewards."
    },
    {
      title: "1. What is RL?",
      notes: "Cover the 6 core terms: Agent, Environment, State, Action, Reward, and Policy. Explain how the agent discovers the best strategy by interacting with the environment."
    },
    {
      title: "2. How RL Works",
      notes: "Walk through the perception-action-reward loop. Show the live GridWorld demo: the agent avoids hazard penalties (-5) and reaches the goal (+10) to maximize total score."
    },
    {
      title: "3. Uniqueness of RL",
      notes: "Highlight the 3 unique traits: (1) Active trial steps vs static data, (2) The Exploration vs Exploitation balance (demo the slider), and (3) Delayed rewards and backwards credit assignment."
    },
    {
      title: "4. Key Advantages",
      notes: "Key advantages: Zero need for human-labeled data, discovery of superhuman strategies (e.g. AlphaGo Move 37), and continuous real-time adaptation."
    },
    {
      title: "5. Real-World Applications",
      notes: "Click through the tabs: Explain RLHF for ChatGPT/Claude alignment, legged robotics, quantitative trading, and DeepMind's data center cooling optimization."
    },
    {
      title: "6. Summary & Takeaways",
      notes: "Recap the main concepts: trial-and-error optimization, self-play, and the role of RL in modern frontier AI models."
    }
  ];

  // --- Application Detail Data ---
  const appData = {
    'app-llm': {
      title: 'LLMs & AI Alignment (RLHF)',
      badge: 'GENERATIVE AI',
      desc: 'Reinforcement Learning from Human Feedback (RLHF) trains large language models to follow instructions, be safe, reduce hallucinations, and self-correct reasoning steps.',
      specs: [
        { label: 'KEY METHODS', val: 'PPO, GRPO, DPO' },
        { label: 'OBJECTIVE', val: 'Helpfulness, Safety & Reasoning' },
        { label: 'EXAMPLES', val: 'ChatGPT, Gemini, Claude, DeepSeek' }
      ],
      detail: 'RL rewards correct reasoning steps, allowing models to search and correct mistakes during problem-solving.'
    },
    'app-robotics': {
      title: 'Robotics & Autonomous Locomotion',
      badge: 'EMBODIED AI',
      desc: 'Robots train in parallel physics simulations (Sim-to-Real) to master walking on rough terrain, recovering balance, and grasping objects.',
      specs: [
        { label: 'KEY METHODS', val: 'PPO, SAC, Sim-to-Real' },
        { label: 'OBJECTIVE', val: 'Stability, Balance & Manipulation' },
        { label: 'EXAMPLES', val: 'Boston Dynamics, Unitree, Quadruped Robots' }
      ],
      detail: 'Sim-to-real transfer trains thousands of virtual robots under varying gravity and friction before real deployment.'
    },
    'app-games': {
      title: 'Gaming & Multi-Agent Simulations',
      badge: 'STRATEGY',
      desc: 'RL agents master board games (Go, Chess) and real-time esports (Dota 2, StarCraft II), discovering moves human experts never conceived.',
      specs: [
        { label: 'KEY METHODS', val: 'AlphaZero, MCTS, Self-Play' },
        { label: 'OBJECTIVE', val: 'Superhuman Win Rate' },
        { label: 'EXAMPLES', val: 'AlphaGo, AlphaZero, OpenAI Five' }
      ],
      detail: 'AlphaZero learned chess in 4 hours from scratch by playing against itself, defeating world-champion engines.'
    },
    'app-finance': {
      title: 'Quantitative Finance & Smart Execution',
      badge: 'FINANCIAL TECH',
      desc: 'Financial algorithms use RL for optimal trade execution and portfolio balancing while minimizing market impact and slippage.',
      specs: [
        { label: 'KEY METHODS', val: 'DDPG, Policy Gradient' },
        { label: 'OBJECTIVE', val: 'Maximize Return & Minimize Risk' },
        { label: 'EXAMPLES', val: 'Algorithmic Trading & Arbitrage' }
      ],
      detail: 'RL agents adapt to changing market volatility without relying on rigid historical assumptions.'
    },
    'app-energy': {
      title: 'Energy Systems & Industrial Operations',
      badge: 'INFRASTRUCTURE',
      desc: 'Industrial facilities and data centers use RL to automate cooling systems and power grid efficiency.',
      specs: [
        { label: 'KEY METHODS', val: 'Model-Based Deep RL' },
        { label: 'OBJECTIVE', val: 'Minimize Energy Consumption' },
        { label: 'EXAMPLES', val: 'Google Data Center Cooling (40% Cut)' }
      ],
      detail: 'DeepMind used RL to control cooling pumps and chillers, reducing Google data center cooling energy by 40%.'
    },
    'app-autonomous': {
      title: 'Self-Driving Vehicles & Navigation',
      badge: 'MOBILITY',
      desc: 'Autonomous vehicles deploy RL for high-level driving decisions, such as highway merging, roundabout navigation, and pedestrian safety.',
      specs: [
        { label: 'KEY METHODS', val: 'Hierarchical RL' },
        { label: 'OBJECTIVE', val: 'Zero Collisions & Smooth Driving' },
        { label: 'EXAMPLES', val: 'Trajectory Planning & Highway Merging' }
      ],
      detail: 'Decouples strategic route decisions from microsecond emergency braking safety controllers.'
    }
  };

  // --- Audio Synthesizer for UI SFX ---
  function playBeep(freq = 440, type = 'sine', duration = 0.05, gain = 0.04) {
    if (!state.isSoundEnabled) return;
    try {
      if (!state.audioCtx) {
        state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (state.audioCtx.state === 'suspended') {
        state.audioCtx.resume();
      }
      const osc = state.audioCtx.createOscillator();
      const gainNode = state.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, state.audioCtx.currentTime);
      gainNode.gain.setValueAtTime(gain, state.audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + duration);
      osc.connect(gainNode);
      gainNode.connect(state.audioCtx.destination);
      osc.start();
      osc.stop(state.audioCtx.currentTime + duration);
    } catch (e) {
      console.debug('Audio notice:', e);
    }
  }

  function playSlideTransitionSound() {
    playBeep(520, 'triangle', 0.06, 0.03);
  }

  function playClickSound() {
    playBeep(880, 'sine', 0.03, 0.02);
  }

  // --- DOM Elements ---
  const slides = document.querySelectorAll('.slide');
  const slideIndicator = document.getElementById('slideIndicator');
  const progressBar = document.getElementById('deckProgressBar');
  const slideDotsContainer = document.getElementById('slideDotsContainer');
  const prevBtn = document.getElementById('prevSlideBtn');
  const nextBtn = document.getElementById('nextSlideBtn');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const overviewToggleBtn = document.getElementById('overviewToggleBtn');
  const overviewModal = document.getElementById('overviewModal');
  const overviewGrid = document.getElementById('overviewGrid');
  const closeOverviewBtn = document.getElementById('closeOverviewBtn');
  const notesToggleBtn = document.getElementById('notesToggleBtn');
  const notesDrawer = document.getElementById('notesDrawer');
  const closeNotesBtn = document.getElementById('closeNotesBtn');
  const notesSlideTitle = document.getElementById('notesSlideTitle');
  const notesBody = document.getElementById('notesBody');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const soundToggleBtn = document.getElementById('soundToggleBtn');
  const soundIcon = document.getElementById('soundIcon');
  const autoplayBtn = document.getElementById('autoplayBtn');
  const autoplayIcon = document.getElementById('autoplayIcon');
  const helpBtn = document.getElementById('helpBtn');
  const shortcutsModal = document.getElementById('shortcutsModal');
  const closeShortcutsBtn = document.getElementById('closeShortcutsBtn');

  // --- Slide Navigation System ---
  function updateDeckUI() {
    // 1. Update Slides Active Class
    slides.forEach((slide, idx) => {
      if (idx === state.currentSlide) {
        slide.classList.add('active');
      } else {
        slide.classList.remove('active');
      }
    });

    document.body.setAttribute('data-active-slide', state.currentSlide);

    // 2. Update Header Slide Indicator (e.g. 01 / 07)
    const currentNumStr = String(state.currentSlide + 1).padStart(2, '0');
    const totalNumStr = String(state.totalSlides).padStart(2, '0');
    slideIndicator.textContent = `${currentNumStr} / ${totalNumStr}`;

    // 3. Update Progress Bar
    const progressPercent = ((state.currentSlide + 1) / state.totalSlides) * 100;
    progressBar.style.width = `${progressPercent}%`;

    // 4. Update Navigation Dots
    const dots = slideDotsContainer.querySelectorAll('.dot-btn');
    dots.forEach((dot, idx) => {
      if (idx === state.currentSlide) {
        dot.classList.add('active');
      } else {
        dot.classList.remove('active');
      }
    });

    // 5. Update Presenter Notes
    const noteData = presenterNotes[state.currentSlide] || {
      title: `Slide ${state.currentSlide + 1}`,
      notes: "No additional notes for this slide."
    };
    notesSlideTitle.textContent = `${currentNumStr}. ${noteData.title}`;
    notesBody.innerHTML = `<p>${noteData.notes}</p>`;

    // 6. Update Overview Grid active card
    const oCards = overviewGrid.querySelectorAll('.overview-card');
    oCards.forEach((c, idx) => {
      if (idx === state.currentSlide) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });
  }

  function goToSlide(index) {
    if (index < 0 || index >= state.totalSlides) return;
    if (state.currentSlide !== index) {
      // Set directional transition
      const dir = index > state.currentSlide ? 'forward' : 'backward';
      document.body.setAttribute('data-nav-dir', dir);

      state.currentSlide = index;
      playSlideTransitionSound();
      updateDeckUI();
    }
  }

  function nextSlide() {
    if (state.currentSlide < state.totalSlides - 1) {
      goToSlide(state.currentSlide + 1);
    } else if (state.isAutoplay) {
      goToSlide(0);
    }
  }

  function prevSlide() {
    if (state.currentSlide > 0) {
      goToSlide(state.currentSlide - 1);
    }
  }

  // --- Build Jump Dots ---
  function buildSlideDots() {
    slideDotsContainer.innerHTML = '';
    slides.forEach((slide, idx) => {
      const dot = document.createElement('button');
      dot.className = 'dot-btn';
      dot.title = `Jump to Slide ${idx + 1}: ${slide.dataset.title || ''}`;
      dot.setAttribute('aria-label', `Slide ${idx + 1}`);
      dot.addEventListener('click', () => {
        goToSlide(idx);
      });
      slideDotsContainer.appendChild(dot);
    });
  }

  // --- Build Overview Grid ---
  function buildOverviewGrid() {
    overviewGrid.innerHTML = '';
    slides.forEach((slide, idx) => {
      const card = document.createElement('div');
      card.className = 'overview-card';
      const numStr = String(idx + 1).padStart(2, '0');
      const titleStr = slide.dataset.title || `Slide ${idx + 1}`;
      card.innerHTML = `
        <span class="o-card-num">SLIDE ${numStr}</span>
        <span class="o-card-title">${titleStr}</span>
      `;
      card.addEventListener('click', () => {
        goToSlide(idx);
        overviewModal.classList.remove('open');
      });
      overviewGrid.appendChild(card);
    });
  }

  // --- Theme Toggle ---
  function toggleTheme() {
    state.isDarkTheme = !state.isDarkTheme;
    if (state.isDarkTheme) {
      document.body.classList.remove('theme-light');
      document.body.classList.add('theme-dark');
    } else {
      document.body.classList.remove('theme-dark');
      document.body.classList.add('theme-light');
    }
    playClickSound();
  }

  // --- Fullscreen Toggle ---
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.warn(`Error attempting fullscreen: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
    playClickSound();
  }

  // --- Autoplay Toggle ---
  function toggleAutoplay() {
    state.isAutoplay = !state.isAutoplay;
    if (state.isAutoplay) {
      autoplayBtn.classList.add('active');
      autoplayBtn.title = "Pause Autoplay (A)";
      autoplayIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
      state.autoplayTimer = setInterval(nextSlide, 7000);
    } else {
      autoplayBtn.classList.remove('active');
      autoplayBtn.title = "Start Autoplay (A)";
      autoplayIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
      clearInterval(state.autoplayTimer);
    }
    playClickSound();
  }

  // --- Sound Toggle ---
  function toggleSound() {
    state.isSoundEnabled = !state.isSoundEnabled;
    if (state.isSoundEnabled) {
      soundIcon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>';
      playClickSound();
    } else {
      soundIcon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>';
    }
  }

  // --- GridWorld Simulation Setup ---
  function initGridWorld() {
    const gridBoard = document.getElementById('gridworld');
    if (!gridBoard) return;

    function renderBoard() {
      gridBoard.innerHTML = '';
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const cell = document.createElement('div');
          cell.className = 'grid-cell';
          cell.dataset.r = r;
          cell.dataset.c = c;

          const isAgent = state.gridworld.agentPos.r === r && state.gridworld.agentPos.c === c;
          const isGoal = state.gridworld.goalPos.r === r && state.gridworld.goalPos.c === c;
          const isHazard = state.gridworld.hazards.some(h => h.r === r && h.c === c);

          if (isAgent) {
            cell.classList.add('agent-here');
            cell.innerHTML = '<span class="cell-tag">AGENT</span>';
          } else if (isGoal) {
            cell.classList.add('goal');
            cell.innerHTML = '<span class="cell-tag">GOAL +10</span>';
          } else if (isHazard) {
            cell.classList.add('hazard');
            cell.innerHTML = '<span class="cell-tag">HAZARD -5</span>';
          } else {
            cell.innerHTML = `<span style="opacity:0.35">${r},${c}</span>`;
          }

          gridBoard.appendChild(cell);
        }
      }

      // Update telemetry
      document.getElementById('simStateDisplay').textContent = `(${state.gridworld.agentPos.r}, ${state.gridworld.agentPos.c})`;
      document.getElementById('simReturnDisplay').textContent = state.gridworld.totalReturn.toFixed(1);
    }

    function stepSim() {
      if (state.gridworld.pathIndex >= state.gridworld.optimalPath.length - 1) {
        state.gridworld.pathIndex = 0;
        state.gridworld.totalReturn = 0;
      } else {
        state.gridworld.pathIndex++;
      }

      const nextStep = state.gridworld.optimalPath[state.gridworld.pathIndex];
      state.gridworld.agentPos = { r: nextStep.r, c: nextStep.c };
      state.gridworld.totalReturn += nextStep.rwd;

      document.getElementById('simActionDisplay').innerHTML = nextStep.a;
      document.getElementById('simRewardDisplay').textContent = nextStep.rwd > 0 ? `+${nextStep.rwd.toFixed(1)}` : nextStep.rwd.toFixed(1);

      playBeep(nextStep.rwd > 0 ? 800 : 400, 'triangle', 0.05, 0.03);
      renderBoard();
    }

    function resetSim() {
      state.gridworld.pathIndex = 0;
      state.gridworld.agentPos = { r: 0, c: 0 };
      state.gridworld.totalReturn = 0;
      if (state.gridworld.isAutoRunning) {
        clearInterval(state.gridworld.autoInterval);
        state.gridworld.isAutoRunning = false;
        document.getElementById('simAutoBtn').textContent = 'AUTO-RUN';
      }
      document.getElementById('simActionDisplay').textContent = 'START';
      document.getElementById('simRewardDisplay').textContent = '0.0';
      renderBoard();
      playClickSound();
    }

    function toggleAutoSim() {
      state.gridworld.isAutoRunning = !state.gridworld.isAutoRunning;
      const autoBtn = document.getElementById('simAutoBtn');
      if (state.gridworld.isAutoRunning) {
        autoBtn.textContent = 'PAUSE';
        autoBtn.classList.add('primary');
        state.gridworld.autoInterval = setInterval(stepSim, 600);
      } else {
        autoBtn.textContent = 'AUTO-RUN';
        autoBtn.classList.remove('primary');
        clearInterval(state.gridworld.autoInterval);
      }
      playClickSound();
    }

    document.getElementById('simStepBtn').addEventListener('click', stepSim);
    document.getElementById('simResetBtn').addEventListener('click', resetSim);
    document.getElementById('simAutoBtn').addEventListener('click', toggleAutoSim);

    renderBoard();
  }

  // --- Multi-Armed Bandit Setup ---
  function initBandit() {
    const banditArmsContainer = document.getElementById('banditArms');
    const epsilonSlider = document.getElementById('epsilonSlider');
    const epsilonValDisplay = document.getElementById('epsilonValDisplay');
    const pullBanditBtn = document.getElementById('pullBanditBtn');
    const banditStats = document.getElementById('banditStats');

    if (!banditArmsContainer || !epsilonSlider) return;

    function renderBanditArms() {
      banditArmsContainer.innerHTML = '';
      for (let i = 0; i < 4; i++) {
        const pulls = state.bandit.pullCounts[i];
        const rew = state.bandit.totalRewards[i];
        const empiricalRate = pulls > 0 ? (rew / pulls).toFixed(2) : '0.00';

        const armCard = document.createElement('div');
        armCard.className = 'bandit-arm-card';
        armCard.id = `bandit-arm-${i}`;
        armCard.innerHTML = `
          <span class="arm-title">SLOT #${i + 1}</span>
          <span class="arm-rate">${empiricalRate}</span>
          <small style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">${pulls} pulls</small>
        `;
        armCard.addEventListener('click', () => {
          pullSingleArm(i);
        });
        banditArmsContainer.appendChild(armCard);
      }
    }

    function pullSingleArm(armIndex) {
      const win = Math.random() < state.bandit.trueRates[armIndex];
      const payout = win ? 1 : 0;
      state.bandit.pullCounts[armIndex]++;
      state.bandit.totalRewards[armIndex] += payout;

      playBeep(win ? 700 : 250, 'square', 0.04, 0.02);

      const armEl = document.getElementById(`bandit-arm-${armIndex}`);
      if (armEl) {
        armEl.classList.add('pulled');
        setTimeout(() => armEl.classList.remove('pulled'), 200);
      }

      banditStats.innerHTML = `Pulled Slot #${armIndex + 1} &rarr; Result: <strong>${win ? '+1.0 WIN' : '+0.0 LOSS'}</strong> | True yield: ${(state.bandit.trueRates[armIndex] * 100).toFixed(0)}%`;
      renderBanditArms();
    }

    function simulateBatchPulls() {
      let exploreCount = 0;
      let exploitCount = 0;

      for (let step = 0; step < 100; step++) {
        let chosenArm = 0;
        if (Math.random() < state.bandit.epsilon) {
          chosenArm = Math.floor(Math.random() * 4);
          exploreCount++;
        } else {
          let bestArm = 0;
          let bestVal = -1;
          for (let a = 0; a < 4; a++) {
            const mean = state.bandit.pullCounts[a] > 0 ? (state.bandit.totalRewards[a] / state.bandit.pullCounts[a]) : 0.5;
            if (mean > bestVal) {
              bestVal = mean;
              bestArm = a;
            }
          }
          chosenArm = bestArm;
          exploitCount++;
        }

        const win = Math.random() < state.bandit.trueRates[chosenArm];
        state.bandit.pullCounts[chosenArm]++;
        state.bandit.totalRewards[chosenArm] += (win ? 1 : 0);
      }

      const totalWins = state.bandit.totalRewards.reduce((a, b) => a + b, 0);
      const totalPulls = state.bandit.pullCounts.reduce((a, b) => a + b, 0);
      const totalWinPct = ((totalWins / totalPulls) * 100).toFixed(1);

      banditStats.innerHTML = `100 Pulls: <strong>${exploitCount} Exploited</strong> vs <strong>${exploreCount} Explored</strong>. Overall Win Rate: <strong>${totalWinPct}%</strong>.`;
      playBeep(650, 'sine', 0.1, 0.04);
      renderBanditArms();
    }

    epsilonSlider.addEventListener('input', (e) => {
      state.bandit.epsilon = parseFloat(e.target.value);
      const eps = state.bandit.epsilon;
      if (eps === 0) {
        epsilonValDisplay.textContent = '100% EXPLOIT (ε = 0.00)';
      } else if (eps === 1) {
        epsilonValDisplay.textContent = '100% EXPLORE (ε = 1.00)';
      } else {
        epsilonValDisplay.textContent = `BALANCED (ε = ${eps.toFixed(2)})`;
      }
    });

    pullBanditBtn.addEventListener('click', simulateBatchPulls);
    renderBanditArms();
  }

  // --- Comparison Tabs Switcher ---
  function initComparisonTabs() {
    const tabs = document.querySelectorAll('.comp-tab');
    const contents = document.querySelectorAll('.comp-content');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetId = tab.dataset.tab;
        tabs.forEach(t => t.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        const targetContent = document.getElementById(targetId);
        if (targetContent) targetContent.classList.add('active');

        playClickSound();
      });
    });
  }

  // --- Application Detail Switcher ---
  function initAppSwitcher() {
    const navItems = document.querySelectorAll('.app-nav-item');
    const detailPane = document.getElementById('appDetailPane');

    function renderAppDetail(appKey) {
      const data = appData[appKey];
      if (!data || !detailPane) return;

      const specsHtml = data.specs.map(s => `
        <div class="spec-box">
          <div class="spec-lbl">${s.label}</div>
          <div class="spec-val">${s.val}</div>
        </div>
      `).join('');

      detailPane.innerHTML = `
        <div class="detail-head">
          <div>
            <span class="mono-badge">${data.badge}</span>
            <h3 class="detail-title" style="margin-top:6px;">${data.title}</h3>
          </div>
        </div>
        <p class="detail-p">${data.desc}</p>
        <div class="detail-specs-grid">
          ${specsHtml}
        </div>
        <div style="background:var(--bg-card);border:1px solid var(--border-subtle);border-radius:8px;padding:12px 14px;font-size:0.85rem;color:var(--text-secondary);margin-top:auto;">
          <strong>Deployment Insight:</strong> ${data.detail}
        </div>
      `;
    }

    navItems.forEach(item => {
      item.addEventListener('click', () => {
        navItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        const appKey = item.dataset.app;
        renderAppDetail(appKey);
        playClickSound();
      });
    });

    // Initial render
    renderAppDetail('app-llm');
  }

  // --- Keyboard Shortcuts Listener ---
  function initKeyboardEvents() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      switch (e.key) {
        case 'ArrowRight':
        case ' ':
        case 'PageDown':
        case 'l':
          e.preventDefault();
          nextSlide();
          break;

        case 'ArrowLeft':
        case 'PageUp':
        case 'h':
        case 'p':
          e.preventDefault();
          prevSlide();
          break;

        case 'Home':
          e.preventDefault();
          goToSlide(0);
          break;

        case 'End':
          e.preventDefault();
          goToSlide(state.totalSlides - 1);
          break;

        case 'o':
        case 'O':
          e.preventDefault();
          overviewModal.classList.toggle('open');
          playClickSound();
          break;

        case 'n':
        case 'N':
          e.preventDefault();
          notesDrawer.classList.toggle('open');
          playClickSound();
          break;

        case 'f':
        case 'F':
          e.preventDefault();
          toggleFullscreen();
          break;

        case 't':
        case 'T':
          e.preventDefault();
          toggleTheme();
          break;

        case 'a':
        case 'A':
          e.preventDefault();
          toggleAutoplay();
          break;

        case '?':
          e.preventDefault();
          shortcutsModal.classList.toggle('open');
          playClickSound();
          break;

        case 'Escape':
          overviewModal.classList.remove('open');
          shortcutsModal.classList.remove('open');
          notesDrawer.classList.remove('open');
          break;

        default:
          const digit = parseInt(e.key, 10);
          if (!isNaN(digit) && digit >= 1 && digit <= state.totalSlides) {
            goToSlide(digit - 1);
          }
          break;
      }
    });
  }

  // --- Modal & Button Listeners ---
  function initEventListeners() {
    prevBtn.addEventListener('click', prevSlide);
    nextBtn.addEventListener('click', nextSlide);
    themeToggleBtn.addEventListener('click', toggleTheme);
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    soundToggleBtn.addEventListener('click', toggleSound);
    autoplayBtn.addEventListener('click', toggleAutoplay);

    overviewToggleBtn.addEventListener('click', () => {
      overviewModal.classList.toggle('open');
      playClickSound();
    });
    closeOverviewBtn.addEventListener('click', () => {
      overviewModal.classList.remove('open');
      playClickSound();
    });

    notesToggleBtn.addEventListener('click', () => {
      notesDrawer.classList.toggle('open');
      playClickSound();
    });
    closeNotesBtn.addEventListener('click', () => {
      notesDrawer.classList.remove('open');
      playClickSound();
    });

    helpBtn.addEventListener('click', () => {
      shortcutsModal.classList.toggle('open');
      playClickSound();
    });
    closeShortcutsBtn.addEventListener('click', () => {
      shortcutsModal.classList.remove('open');
      playClickSound();
    });

    overviewModal.addEventListener('click', (e) => {
      if (e.target === overviewModal) overviewModal.classList.remove('open');
    });
    shortcutsModal.addEventListener('click', (e) => {
      if (e.target === shortcutsModal) shortcutsModal.classList.remove('open');
    });

    // Touch gesture swipe support
    let touchStartX = 0;
    let touchEndX = 0;
    const viewport = document.getElementById('deck-container');

    viewport.addEventListener('touchstart', (e) => {
      touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    viewport.addEventListener('touchend', (e) => {
      touchEndX = e.changedTouches[0].screenX;
      if (touchEndX < touchStartX - 50) {
        nextSlide();
      } else if (touchEndX > touchStartX + 50) {
        prevSlide();
      }
    }, { passive: true });
  }

  // --- Global expose for inline HTML onclick handlers ---
  window.goToSlide = goToSlide;
  window.isDeckSoundMuted = () => !state.isSoundEnabled;

  // --- Initialize All Subsystems ---
  function init() {
    buildSlideDots();
    buildOverviewGrid();
    initGridWorld();
    initBandit();
    initComparisonTabs();
    initAppSwitcher();
    initKeyboardEvents();
    initEventListeners();
    updateDeckUI();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
