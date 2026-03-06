// === Anima Training UI — Client ===

const DEFAULT_NEGATIVE_PROMPT = 'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, low quality, worst quality, blurry, bad anatomy, extra limbs, deformed, watermark, text, signature, bareness, artifacts, hands, copyrights name, jpeg_artifacts, scan_artifacts, bad hands, missing fingers, extra digit, fewer digits, artistic error, ye-pop, deviantart, logo, patreon logo';

let currentJob = null;
let ws = null;
let isDirty = false;
let lastSavedConfig = null;
let lastSavedDataset = null;
let lastSavedPrompts = [];
let lastSavedNegativePrompt = '';
let samplesPollTimer = null;
let isDraggingBg = false;
let bgPosPercent = { x: 50, y: 50 };

// --- DOM Refs ---
const $ = (id) => document.getElementById(id);
const jobListEl = $('job-list');
const emptyState = $('empty-state');
const jobEditor = $('job-editor');
const jobTitle = $('job-title');
const consoleOutput = $('console-output');

// ==========================================
//  API
// ==========================================

// Deletion API
async function deleteSamples(paths) {
    if (!currentJob) return;
    try {
        for (const fullPath of paths) {
            // Path is /api/jobs/:name/samples/samples/filename.png
            const parts = fullPath.split('/samples/');
            const relPath = parts[parts.length - 1];

            await fetch(`/api/jobs/${currentJob}/samples/${relPath}`, {
                method: 'DELETE'
            });
        }

        // Remove from local state
        paths.forEach(p => sampleState.selectedPaths.delete(p));

        // Refresh UI
        loadSamples();
    } catch (err) {
        console.error("Delete failed", err);
        showToast("Error deleting samples", "danger");
    }
}
async function api(url, opts = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
        body: opts.body ? JSON.stringify(opts.body) : undefined
    });
    if (!res.ok) {
        let msg = res.statusText;
        try { const err = await res.json(); msg = err.error || msg; } catch (_) { }
        throw new Error(msg);
    }
    return res.json();
}

// ==========================================
//  WebSocket
// ==========================================

function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}`);

    ws.onopen = () => {
        if (currentJob) {
            ws.send(JSON.stringify({ type: 'subscribe', job: currentJob }));
        }
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.job !== currentJob) return;

            if (msg.type === 'log') {
                appendConsole(msg.data);
            } else if (msg.type === 'status') {
                if (msg.data === 'generating') return; // Ignore generation status for Training button
                updateRunningState(msg.data === 'running');
            }
        } catch (e) { }
    };

    ws.onclose = () => {
        setTimeout(connectWS, 3000);
    };
}

function subscribeToJob(jobName) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'subscribe', job: jobName }));
    }
}

// ==========================================
//  Job List
// ==========================================

async function loadJobs() {
    const jobs = await api('/api/jobs');
    jobListEl.innerHTML = '';

    if (jobs.length === 0) {
        jobListEl.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted)">No jobs yet</div>';
        return;
    }

    jobs.forEach(job => {
        const el = document.createElement('div');
        el.className = `job-item${job.name === currentJob ? ' active' : ''}${job.running ? ' running' : ''}`;
        el.innerHTML = `
            <div class="status-dot"></div>
            <span class="job-name">${job.name}</span>
        `;
        el.addEventListener('click', () => selectJob(job.name));
        jobListEl.appendChild(el);
    });
}

async function selectJob(name) {
    if (currentJob) savePromptTransientSettings();
    if (isDirty && !confirm('Unsaved changes. Switch anyway?')) return;
    isDirty = false;

    currentJob = name;
    if (samplesPollTimer) { clearInterval(samplesPollTimer); samplesPollTimer = null; }
    localStorage.setItem('lastJob', name);
    jobTitle.textContent = name;
    emptyState.classList.add('hidden');
    jobEditor.classList.remove('hidden');

    try {
        // Load available GPUs
        await loadGPUs();
        await loadGenGPUs();

        // Load job data
        const data = await api(`/api/jobs/${name}`);
        populateConfig(data.config);
        populateDataset(data.dataset);

        // Load prompts
        await loadPrompts();

        // Check run status
        const status = await api(`/api/jobs/${name}/train/status`);
        updateRunningState(status.running);

        // Set default negative prompt if no saved value exists for this job
        const savedTransient = localStorage.getItem(`prompt_transient_${name}`);
        if (!savedTransient || !JSON.parse(savedTransient).negative_prompt) {
            $('global-negative-prompt').value = DEFAULT_NEGATIVE_PROMPT;
        }

        // Save initial state for dirty checking
        lastSavedConfig = JSON.parse(JSON.stringify(gatherConfig()));
        lastSavedDataset = JSON.parse(JSON.stringify(gatherDataset()));
        lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
        lastSavedNegativePrompt = $('global-negative-prompt').value;


        // Subscribe WS
        subscribeToJob(name);

        // Reset console
        consoleOutput.textContent = 'Waiting for training to start...';

        // Reset save button
        $('btn-save').classList.add('hidden');
        $('btn-discard').classList.add('hidden');

        // Refresh job list highlight
        loadJobs();

        // Load samples
        loadSamples();
        loadCheckpoints();
        loadPromptTransientSettings();
    } catch (err) {
        console.error(`Failed to load job "${name}":`, err);
        showToast(`Failed to load job: ${err.message}`, 'danger');
        currentJob = null;
        localStorage.removeItem('lastJob');
        jobEditor.classList.add('hidden');
        emptyState.classList.remove('hidden');
    }
}

function updateRunningState(running) {
    $('btn-run').classList.toggle('hidden', running);
    $('btn-stop').classList.toggle('hidden', !running);

    // Update sidebar dot
    document.querySelectorAll('.job-item').forEach(el => {
        const name = el.querySelector('.job-name').textContent;
        if (name === currentJob) {
            el.classList.toggle('running', running);
        }
    });
}

// ==========================================
//  Config ↔ UI Mapping
// ==========================================

function populateConfig(config) {
    const t = config.training_arguments || {};
    const n = config.network_arguments || {};
    const a = config.anima_arguments || {};

    // Training
    $('cfg-learning-rate').value = t.learning_rate || '5e-5';
    $('cfg-text-encoder-lr').value = t.text_encoder_lr || '5e-5';
    $('cfg-optimizer').value = t.optimizer_type || 'AdamW8bit';
    $('cfg-lr-scheduler').value = t.lr_scheduler || 'cosine';
    $('cfg-lr-warmup').value = t.lr_warmup_steps ?? 100;
    $('cfg-seed').value = t.seed ?? 42;

    // Extract weight decay
    let wdValue = '0';
    let decoupleValue = true;
    if (t.optimizer_args && Array.isArray(t.optimizer_args)) {
        const wdArg = t.optimizer_args.find(arg => String(arg).startsWith('weight_decay='));
        if (wdArg) {
            wdValue = wdArg.split('=')[1];
        }
        const decoupleArg = t.optimizer_args.find(arg => String(arg).startsWith('decouple='));
        if (decoupleArg) {
            decoupleValue = decoupleArg.split('=')[1].toLowerCase() === 'true';
        }
    }
    $('cfg-weight-decay').value = wdValue;
    $('cfg-decouple').checked = decoupleValue;

    // Update conditional visibility
    updateOptimizerOptions();


    const maxSteps = t.max_train_steps;
    const isSteps = maxSteps && maxSteps > 0;
    document.querySelector(`input[name="duration-unit"][value="${isSteps ? 'steps' : 'epochs'}"]`).checked = true;
    updateDurationUnit();

    $('cfg-max-epochs').value = t.max_train_epochs ?? 20;
    $('cfg-save-every').value = t.save_every_n_epochs ?? 1;

    $('cfg-max-steps').value = t.max_train_steps ?? 1000;
    $('cfg-save-every-steps').value = t.save_every_n_steps ?? 500;

    $('cfg-output-name').value = t.output_name || 'my_anima_lora';
    $('cfg-save-format').value = t.save_model_as || 'safetensors';
    $('cfg-save-precision').value = t.save_precision || 'bf16';
    $('cfg-mixed-precision').value = t.mixed_precision || 'bf16';
    $('cfg-workers').value = t.max_data_loader_n_workers ?? 4;
    $('cfg-grad-acc').value = t.gradient_accumulation_steps ?? 1;
    $('cfg-gradient-checkpointing').checked = t.gradient_checkpointing ?? true;
    $('cfg-flash-attn').checked = t.flash_attn ?? false;
    $('cfg-torch-compile').checked = t.torch_compile ?? false;
    $('cfg-lowram').checked = t.lowram ?? false;
    $('cfg-blocks-to-swap').value = t.blocks_to_swap ?? 0;
    $('cfg-persistent-workers').checked = t.persistent_data_loader_workers ?? true;
    $('cfg-cache-latents').checked = t.cache_latents_to_disk ?? true;
    $('cfg-vae-batch').value = t.vae_batch_size ?? 1;
    $('cfg-cache-te').checked = t.cache_text_encoder_outputs_to_disk ?? true;

    // Check GPU boxes based on config
    const savedIds = (config.gpu_ids || '').split(',').map(s => s.trim()).filter(s => s);
    document.querySelectorAll('input[name="gpu-select"]').forEach(cb => {
        cb.checked = savedIds.length === 0 || savedIds.includes(cb.value);
        // Also update card class if parent exists
        const card = cb.closest('.gpu-card');
        if (card) card.classList.toggle('selected', cb.checked);
    });

    const sampleInterval = t.sample_every_n_epochs;
    const sampleIntervalSteps = t.sample_every_n_steps;
    const enableSampling = (sampleInterval !== null && sampleInterval > 0) || (sampleIntervalSteps !== null && sampleIntervalSteps > 0);

    $('cfg-enable-sampling').checked = enableSampling;
    $('cfg-sample-every').value = sampleInterval || 1;
    $('cfg-sample-every-steps').value = sampleIntervalSteps || 100;
    $('group-sample-every').classList.toggle('hidden', !enableSampling);

    // Anima
    $('cfg-timestep-method').value = a.timestep_sample_method || 'logit_normal';
    $('cfg-flow-shift').value = a.discrete_flow_shift ?? 3.0;

    // Network
    $('cfg-network-module').value = n.network_module || 'networks.lora_anima';
    $('cfg-network-dim').value = n.network_dim ?? 16;
    $('cfg-network-alpha').value = n.network_alpha ?? 16;
    $('cfg-unet-only').checked = n.network_train_unet_only ?? true;
    $('cfg-network-weights').value = n.network_weights || '';
    $('cfg-resume').value = n.resume || '';
}

function populateDataset(dataset) {
    const g = dataset.general || {};
    const d = (dataset.datasets && dataset.datasets[0]) || {};
    const s = (d.subsets && d.subsets[0]) || {};

    $('cfg-image-dir').value = s.image_dir || '';
    const res = Array.isArray(d.resolution) ? d.resolution[0] : (d.resolution || 1536);
    $('cfg-resolution').value = res;
    $('cfg-batch-size').value = d.batch_size ?? 4;
    $('cfg-caption-ext').value = d.caption_extension || '.txt';
    $('cfg-num-repeats').value = s.num_repeats ?? 1;
    $('cfg-keep-tokens').value = s.keep_tokens ?? 1;
    $('cfg-flip-aug').checked = s.flip_aug ?? false;
    $('cfg-caption-prefix').value = s.caption_prefix || '';

    // Caption Settings
    $('cfg-caption-dropout').value = s.caption_dropout_rate ?? 0.05;
    $('cfg-tag-dropout').value = s.caption_tag_dropout_rate ?? 0.0;
    $('cfg-dropout-every-n').value = s.caption_dropout_every_n_epochs ?? 0;
    $('cfg-shuffle-caption').checked = s.shuffle_caption ?? false;

    $('cfg-enable-bucket').checked = g.enable_bucket ?? true;
    $('cfg-bucket-no-upscale').checked = g.bucket_no_upscale ?? true;
    $('cfg-min-bucket').value = g.min_bucket_reso ?? 512;
    $('cfg-max-bucket').value = g.max_bucket_reso ?? 1536;
    $('cfg-bucket-steps').value = g.bucket_reso_steps ?? 64;
}

function updateOptimizerOptions() {
    const optimizer = $('cfg-optimizer').value;
    const isProdigy = optimizer.includes('Prodigy') || optimizer.includes('DAdapt');
    $('group-decouple').classList.toggle('hidden', !isProdigy);
}

// Helpers for safe parsing
function safeInt(val, fallback = 0) {
    if (val === '' || val === null || val === undefined) return fallback;
    const p = parseInt(val);
    return isNaN(p) ? fallback : p;
}

function safeFloat(val, fallback = 0.0) {
    if (val === '' || val === null || val === undefined) return fallback;
    const p = parseFloat(val);
    return isNaN(p) ? fallback : p;
}


function gatherConfig() {
    const unit = document.querySelector('input[name="duration-unit"]:checked').value;
    const isEpochs = unit === 'epochs';
    const enableSampling = $('cfg-enable-sampling').checked;

    const optimizerArgs = [];
    const wdValue = $('cfg-weight-decay').value;
    if (wdValue !== '') {
        optimizerArgs.push(`weight_decay=${wdValue}`);
    }

    if (!$('group-decouple').classList.contains('hidden')) {
        const isDecoupled = $('cfg-decouple').checked;
        optimizerArgs.push(`decouple=${isDecoupled ? 'True' : 'False'}`);
    }

    const config = {
        training_arguments: {
            output_name: $('cfg-output-name').value,
            save_model_as: $('cfg-save-format').value,

            max_train_epochs: isEpochs ? safeInt($('cfg-max-epochs').value) : undefined,
            save_every_n_epochs: isEpochs ? safeInt($('cfg-save-every').value) : undefined,
            sample_every_n_epochs: (isEpochs && enableSampling) ? safeInt($('cfg-sample-every').value) : undefined,

            max_train_steps: !isEpochs ? safeInt($('cfg-max-steps').value) : undefined,
            save_every_n_steps: !isEpochs ? safeInt($('cfg-save-every-steps').value) : undefined,
            sample_every_n_steps: (!isEpochs && enableSampling) ? safeInt($('cfg-sample-every-steps').value) : undefined,

            log_with: 'tensorboard',
            learning_rate: safeFloat($('cfg-learning-rate').value),
            text_encoder_lr: safeFloat($('cfg-text-encoder-lr').value),

            optimizer_type: $('cfg-optimizer').value,
            optimizer_args: optimizerArgs.length > 0 ? optimizerArgs : undefined,
            lr_scheduler: $('cfg-lr-scheduler').value,
            lr_warmup_steps: safeInt($('cfg-lr-warmup').value),
            // Hardware
            mixed_precision: $('cfg-mixed-precision').value,
            save_precision: $('cfg-save-precision').value || undefined,
            max_data_loader_n_workers: safeInt($('cfg-workers').value),
            gradient_accumulation_steps: safeInt($('cfg-grad-acc').value),

            max_grad_norm: 1.0,
            gradient_checkpointing: $('cfg-gradient-checkpointing').checked,
            flash_attn: $('cfg-flash-attn').checked,
            torch_compile: $('cfg-torch-compile').checked,
            lowram: $('cfg-lowram').checked,
            blocks_to_swap: safeInt($('cfg-blocks-to-swap').value),
            persistent_data_loader_workers: $('cfg-persistent-workers').checked,
            seed: safeInt($('cfg-seed').value),
            cache_latents_to_disk: $('cfg-cache-latents').checked,
            vae_batch_size: safeInt($('cfg-vae-batch').value),
            cache_text_encoder_outputs_to_disk: $('cfg-cache-te').checked
        },
        network_arguments: {
            network_module: $('cfg-network-module').value,
            network_dim: safeInt($('cfg-network-dim').value),
            network_alpha: safeInt($('cfg-network-alpha').value),
            network_train_unet_only: $('cfg-unet-only').checked,
            ...(($('cfg-network-weights').value) && { network_weights: $('cfg-network-weights').value }),
            ...(($('cfg-resume').value) && { resume: $('cfg-resume').value })
        },
        anima_arguments: {
            timestep_sample_method: $('cfg-timestep-method').value,
            discrete_flow_shift: safeFloat($('cfg-flow-shift').value),

            weighting_scheme: 'logit_normal'
        },
        gpu_ids: Array.from(document.querySelectorAll('input[name="gpu-select"]:checked'))
            .map(cb => cb.value)
            .join(',')
    };
    return config;
}

function gatherDataset() {
    const res = safeInt($('cfg-resolution').value);
    return {
        general: {
            enable_bucket: $('cfg-enable-bucket').checked,
            bucket_no_upscale: $('cfg-bucket-no-upscale').checked,
            min_bucket_reso: safeInt($('cfg-min-bucket').value),
            max_bucket_reso: safeInt($('cfg-max-bucket').value),
            bucket_reso_steps: safeInt($('cfg-bucket-steps').value)
        },
        datasets: [{
            resolution: [res, res],
            batch_size: safeInt($('cfg-batch-size').value),
            caption_extension: $('cfg-caption-ext').value,
            subsets: [{
                image_dir: $('cfg-image-dir').value,
                num_repeats: safeInt($('cfg-num-repeats').value),
                keep_tokens: safeInt($('cfg-keep-tokens').value),
                flip_aug: $('cfg-flip-aug').checked,
                caption_prefix: $('cfg-caption-prefix').value,

                // Caption Settings
                caption_dropout_rate: safeFloat($('cfg-caption-dropout').value),
                caption_tag_dropout_rate: safeFloat($('cfg-tag-dropout').value),
                caption_dropout_every_n_epochs: safeInt($('cfg-dropout-every-n').value),

                shuffle_caption: $('cfg-shuffle-caption').checked
            }]
        }]
    };
}

// ==========================================
//  Save
// ==========================================

async function saveJob() {
    if (!currentJob) return;
    const config = gatherConfig();
    const dataset = gatherDataset();

    // Save Config & Dataset
    await api(`/api/jobs/${currentJob}`, {
        method: 'PUT',
        body: { config, dataset }
    });

    // Save Prompts
    await savePrompts();

    // Update last saved state
    lastSavedConfig = JSON.parse(JSON.stringify(config));
    lastSavedDataset = JSON.parse(JSON.stringify(dataset));
    lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
    lastSavedNegativePrompt = $('global-negative-prompt').value;

    checkDirty();
    showToast('Job saved');
}

function checkDirty() {
    if (!currentJob) return;

    const currentConfig = gatherConfig();
    const currentDataset = gatherDataset();

    // Deep compare
    const configChanged = JSON.stringify(currentConfig) !== JSON.stringify(lastSavedConfig);
    const datasetChanged = JSON.stringify(currentDataset) !== JSON.stringify(lastSavedDataset);
    const promptsChanged = JSON.stringify(currentPrompts) !== JSON.stringify(lastSavedPrompts);
    const negPromptChanged = ($('global-negative-prompt').value || '') !== (lastSavedNegativePrompt || '');

    isDirty = configChanged || datasetChanged || promptsChanged || negPromptChanged;

    if (isDirty) {
        $('btn-save').classList.remove('hidden');
        $('btn-discard').classList.remove('hidden');
    } else {
        $('btn-save').classList.add('hidden');
        $('btn-discard').classList.add('hidden');
    }
}

function discardChanges() {
    if (!currentJob || !isDirty) return;

    showConfirm('Discard Changes', 'Discard all unsaved changes and revert to last saved state?', () => {
        populateConfig(lastSavedConfig);
        populateDataset(lastSavedDataset);
        currentPrompts = JSON.parse(JSON.stringify(lastSavedPrompts));
        renderPrompts();

        isDirty = false;
        $('btn-save').classList.add('hidden');
        $('btn-discard').classList.add('hidden');
        showToast('Changes discarded');
    });
}

// Mark dirty on any input change
document.addEventListener('input', (e) => {
    if (e.target.closest('.tab-content') && e.target.closest('.tab-pane')) {
        checkDirty();
    }
});


// ==========================================
//  Prompts
// ==========================================

let currentPrompts = []; // Array of objects { text, w, h, s, l, d }

async function loadPrompts() {
    if (!currentJob) return;
    const data = await api(`/api/jobs/${currentJob}/prompts`);
    // Parse strings into objects
    currentPrompts = (data.prompts || []).map(line => parsePromptLine(line));
    renderPrompts();
}

function parsePromptLine(line) {
    // Defaults
    const p = { text: '', w: 832, h: 1216, s: 20, l: 7.5, d: 1, skip: false };

    // Check if skipped
    if (line.trim().startsWith('#')) {
        p.skip = true;
        line = line.trim().substring(1).trim();
    }

    // Extract params
    const paramRegex = /\s+--([whdsl])\s+(\S+)/g;
    let match;
    while ((match = paramRegex.exec(line)) !== null) {
        const val = match[2];
        if (match[1] === 'w') p.w = parseInt(val);
        if (match[1] === 'h') p.h = parseInt(val);
        if (match[1] === 's') p.s = parseInt(val);
        if (match[1] === 'd') p.d = parseInt(val);
        if (match[1] === 'l') p.l = parseFloat(val);
    }

    // Extract text (strip out specific params and the negative prompt string)
    p.text = line
        .replace(/\s+--n\s+.*$/i, '') // Remove global negative prompt and everything after it
        .replace(/\s+--[whdsl]\s+\S+/gi, '') // Remove regular parameter flags
        .trim();
    return p;
}

function serializePrompt(p) {
    // Reconstruct line, ensuring no newlines break the backend parsing parser
    const safeText = p.text.replace(/[\r\n]+/g, ' ').trim();
    let line = `${safeText} --w ${p.w} --h ${p.h} --s ${p.s} --d ${p.d} --l ${p.l}`;

    // Append global negative prompt without newlines
    const neg = $('global-negative-prompt').value.replace(/[\r\n]+/g, ' ').trim();
    if (neg) {
        line += ` --n ${neg}`;
    }

    return p.skip ? `# ${line}` : line;
}

async function savePrompts() {
    // Filter out prompts that have no text before saving
    const validPrompts = currentPrompts.filter(p => p.text && p.text.trim().length > 0);
    const lines = validPrompts.map(serializePrompt);

    await api(`/api/jobs/${currentJob}/prompts`, {
        method: 'PUT',
        body: { prompts: lines }
    });
}

function renderPrompts() {
    const list = $('prompts-list');
    const empty = $('prompts-empty');

    if (currentPrompts.length === 0) {
        list.classList.add('hidden');
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = '';

    currentPrompts.forEach((p, idx) => {
        const card = document.createElement('div');
        card.className = `prompt-card-edit${p.skip ? ' skipped' : ''}`;
        card.innerHTML = `
            <div class="prompt-card-header">
                <label class="skip-label">
                    <input type="checkbox" class="p-skip" ${p.skip ? 'checked' : ''}> Skip
                </label>
            </div>
            <textarea class="p-text" rows="2" placeholder="Enter prompt text...">${escapeHTML(p.text)}</textarea>
            <div class="prompt-card-row">
                <div class="compact-input">
                    <label>W</label>
                    <input type="number" class="p-w" value="${p.w}" step="64">
                </div>
                <div class="compact-input">
                    <label>H</label>
                    <input type="number" class="p-h" value="${p.h}" step="64">
                </div>
                <div class="compact-input">
                    <label>Steps</label>
                    <input type="number" class="p-s" value="${p.s}">
                </div>
                <div class="compact-input">
                    <label>Scale</label>
                    <input type="number" class="p-l" value="${p.l}" step="0.5">
                </div>
                <div class="compact-input">
                    <label>Seed</label>
                    <input type="number" class="p-d" value="${p.d}">
                </div>
                <button class="btn btn-ghost btn-sm btn-delete-prompt" title="Delete">🗑️</button>
            </div>
        `;

        // Bind events
        const updateState = () => {
            p.skip = card.querySelector('.p-skip').checked;
            p.text = card.querySelector('.p-text').value;
            p.w = parseInt(card.querySelector('.p-w').value);
            p.h = parseInt(card.querySelector('.p-h').value);
            p.s = parseInt(card.querySelector('.p-s').value);
            p.l = parseFloat(card.querySelector('.p-l').value);
            p.d = parseInt(card.querySelector('.p-d').value);

            card.classList.toggle('skipped', p.skip);

            checkDirty();

        };


        const tx = card.querySelector('.p-text');
        const autoResize = () => {
            tx.style.height = 'auto';
            tx.style.height = (tx.scrollHeight + 2) + 'px';
        };
        tx.addEventListener('input', autoResize);
        // Initial resize
        setTimeout(autoResize, 1);

        card.querySelectorAll('input, textarea').forEach(el => {
            el.addEventListener('input', updateState);
        });

        card.querySelector('.btn-delete-prompt').addEventListener('click', () => deletePrompt(idx));

        list.appendChild(card);
    });
}

function deletePrompt(idx) {
    currentPrompts.splice(idx, 1);
    renderPrompts();
    checkDirty();
}


function addPrompt() {

    // Get defaults from global bar
    const w = parseInt($('global-w').value) || 832;
    const h = parseInt($('global-h').value) || 1216;
    const s = parseInt($('global-s').value) || 28;
    const l = parseFloat($('global-l').value) || 3.5;
    let d = parseInt($('global-d').value);
    // If global seed is 0 or empty, randomize for the new prompt
    if (!d || d === 0) {
        d = Math.floor(Math.random() * 99999) + 1;
    }

    currentPrompts.push({ text: '', w, h, s, l, d, skip: false });
    renderPrompts();
    checkDirty();
}

function applyGlobalSettings() {

    const w = parseInt($('global-w').value);
    const h = parseInt($('global-h').value);
    const s = parseInt($('global-s').value);
    const l = parseFloat($('global-l').value);
    const d = parseInt($('global-d').value);

    currentPrompts.forEach(p => {
        if (w) p.w = w;
        if (h) p.h = h;
        if (s) p.s = s;
        if (l) p.l = l;

        // Seed handling: 0 = random for each prompt, non-zero = apply same seed to all
        if (d === 0) {
            p.d = Math.floor(Math.random() * 99999) + 1; // Random seed 1-99999
        } else if (d) {
            p.d = d;
        }
    });

    renderPrompts();
    renderPrompts();
    checkDirty();
    showToast(d === 0 ? 'Random seeds applied to all prompts' : 'Global settings applied to all prompts');

}

// Helper to escape HTML for textarea
function escapeHTML(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

// === Prompt Tab Persistence ===

function savePromptTransientSettings() {
    if (!currentJob) return;
    const settings = {
        lora_mul: $('gen-lora-mul').value,
        keep_loaded: $('chk-keep-loaded').checked,
        flash_attn: $('gen-flash-attn').checked,
        sage_attn: $('gen-sage-attn').checked,
        global_w: $('global-w').value,
        global_h: $('global-h').value,
        global_s: $('global-s').value,
        global_l: $('global-l').value,
        global_d: $('global-d').value,
        selected_lora: $('gen-lora-select').value,
        negative_prompt: $('global-negative-prompt').value,
        gen_gpu_ids: getSelectedGenGPUs(),
        gen_multi_gpu_mode: $('gen-multi-gpu-mode').value
    };
    localStorage.setItem(`prompt_transient_${currentJob}`, JSON.stringify(settings));
}

function loadPromptTransientSettings() {
    if (!currentJob) return;
    const data = localStorage.getItem(`prompt_transient_${currentJob}`);
    if (!data) return;

    try {
        const settings = JSON.parse(data);
        if (settings.lora_mul !== undefined) $('gen-lora-mul').value = settings.lora_mul;
        if (settings.keep_loaded !== undefined) $('chk-keep-loaded').checked = settings.keep_loaded;
        if (settings.flash_attn !== undefined) $('gen-flash-attn').checked = settings.flash_attn;
        if (settings.sage_attn !== undefined) $('gen-sage-attn').checked = settings.sage_attn;
        if (settings.global_w !== undefined) $('global-w').value = settings.global_w;
        if (settings.global_h !== undefined) $('global-h').value = settings.global_h;
        if (settings.global_s !== undefined) $('global-s').value = settings.global_s;
        if (settings.global_l !== undefined) $('global-l').value = settings.global_l;
        if (settings.global_d !== undefined) $('global-d').value = settings.global_d;
        if (settings.negative_prompt !== undefined) $('global-negative-prompt').value = settings.negative_prompt;
        // Restore gen GPU selection
        if (settings.gen_gpu_ids !== undefined) {
            restoreGenGPUSelection(settings.gen_gpu_ids);
        }
        if (settings.gen_multi_gpu_mode !== undefined) {
            $('gen-multi-gpu-mode').value = settings.gen_multi_gpu_mode;
        }
        // selected_lora is handled in loadCheckpoints
    } catch (e) { }
}

// ==========================================
//  Console
// ==========================================

function appendConsole(text) {
    if (!text) return;

    if (consoleOutput.textContent.startsWith('Waiting')) {
        consoleOutput.textContent = '';
    }

    const wasNearBottom = consoleOutput.scrollHeight - consoleOutput.scrollTop - consoleOutput.clientHeight < 100;

    // Standard Terminal logic: \r overwrites the CURRENT line.
    // We split current content and process the last line surgically.
    let fullText = consoleOutput.textContent + text;

    if (fullText.includes('\r')) {
        const lines = fullText.split('\n');
        const lastLine = lines[lines.length - 1];

        if (lastLine.includes('\r')) {
            const parts = lastLine.split('\r');
            let processedLine = "";
            for (let i = 0; i < parts.length; i++) {
                // If there is content after this \r, it overwrites what was before it.
                // If this is the very last part of the string, we keep it.
                if (i === parts.length - 1) {
                    processedLine += parts[i];
                } else if (parts[i + 1].length > 0) {
                    processedLine = ""; // Overwrite triggered by following content
                } else {
                    processedLine = parts[i]; // Keep until something actually follows the \r
                }
            }
            lines[lines.length - 1] = processedLine;
            fullText = lines.join('\n');
        }
    }

    consoleOutput.textContent = fullText;

    if (wasNearBottom) {
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
}

// ==========================================
//  Samples
// ==========================================

async function loadCheckpoints() {
    if (!currentJob) return;
    const jobAtStart = currentJob;
    const files = await api(`/api/jobs/${currentJob}/checkpoints`);
    if (currentJob !== jobAtStart) return; // job changed while fetching
    const select = $('gen-lora-select');

    // Save current selection
    const currentVal = select.value;

    select.innerHTML = '<option value="">Base Model (No LoRA)</option>';

    files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = `${f.name} (${new Date(f.mtime).toLocaleString()})`;
        select.appendChild(opt);
    });

    // Restore selection if exists
    const data = localStorage.getItem(`prompt_transient_${currentJob}`);
    let savedLora = null;
    if (data) {
        try { savedLora = JSON.parse(data).selected_lora; } catch (e) { }
    }

    const valToRestore = currentVal || savedLora;
    if (valToRestore && Array.from(select.options).some(o => o.value === valToRestore)) {
        select.value = valToRestore;
    }
}

// Sample State
let sampleState = {
    selectedPaths: new Set(),
    lastSelectedPath: null,
    groups: {}, // enum -> [images]
    allImages: [], // flat list for index lookup
    isExplicitMultiSelect: false
};

async function loadSamples(isUpdate = false) {
    if (!currentJob) return;
    const images = await api(`/api/jobs/${currentJob}/samples`);
    const container = $('samples-grid');
    const empty = $('samples-empty');

    if (!images || images.length === 0) {
        if (!isUpdate) {
            container.classList.add('hidden');
            empty.classList.remove('hidden');
        }
        return;
    }

    empty.classList.add('hidden');
    container.classList.remove('hidden');

    // Load manual order from localStorage
    const savedOrder = loadManualOrder();
    const orderMap = new Map();
    if (savedOrder) {
        savedOrder.forEach((item, index) => {
            orderMap.set(item.path, { group: item.group, index: index });
        });
    }

    // 1. Group Images
    const groups = {};
    images.forEach(img => {
        let groupKey;
        if (orderMap.has(img.path)) {
            groupKey = orderMap.get(img.path).group;
        } else {
            const match = img.name.match(/_(\d{2,})_\d{14}/);
            groupKey = match ? match[1] : 'default';
        }

        if (!groups[groupKey]) groups[groupKey] = [];
        groups[groupKey].push(img);
    });

    sampleState.groups = groups;
    sampleState.allImages = images;

    // 2. Render Groups
    container.innerHTML = '';

    const sortedGroupKeys = Object.keys(groups).sort((a, b) => {
        if (a === 'default') return 1;
        if (b === 'default') return -1;
        return parseInt(a) - parseInt(b);
    });

    sortedGroupKeys.forEach(key => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'sample-group';
        groupDiv.dataset.group = key;

        const header = document.createElement('div');
        header.className = 'group-header';
        header.textContent = key === 'default' ? 'Uncategorized' : `Prompt ${parseInt(key) + 1}`;

        const gridDiv = document.createElement('div');
        gridDiv.className = 'group-grid';
        gridDiv.addEventListener('dragover', handleDragOver);
        gridDiv.addEventListener('drop', handleDrop);

        // Sort images in group. 
        // If they have a saved index, use it. Otherwise, use mtime (newest first).
        groups[key].sort((a, b) => {
            const orderA = orderMap.get(a.path);
            const orderB = orderMap.get(b.path);

            if (orderA && orderB) return orderA.index - orderB.index;
            if (orderA) return 1; // Saved items come after new items? 
            if (orderB) return -1;

            return b.mtime - a.mtime; // Default newest first for items without saved order
        });

        groups[key].forEach(img => {
            createSampleCard(img, gridDiv);
        });

        groupDiv.appendChild(header);
        groupDiv.appendChild(gridDiv);
        container.appendChild(groupDiv);
    });

    if (!window._samplesInitialized) {
        initSampleInteractions();
        window._samplesInitialized = true;
    }

    updateSelectionVisuals();
}

function saveManualOrder() {
    if (!currentJob) return;
    const order = [];
    document.querySelectorAll('.sample-group').forEach(group => {
        const groupKey = group.dataset.group;
        group.querySelectorAll('.sample-card').forEach(card => {
            order.push({
                path: card.dataset.path,
                group: groupKey
            });
        });
    });
    localStorage.setItem(`sample_order_${currentJob}`, JSON.stringify(order));
}

function loadManualOrder() {
    if (!currentJob) return null;
    const data = localStorage.getItem(`sample_order_${currentJob}`);
    try {
        return data ? JSON.parse(data) : null;
    } catch (e) {
        return null;
    }
}

function createSampleCard(img, container) {
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.draggable = true;
    card.dataset.path = img.path;
    card.dataset.name = img.name;
    card.dataset.mtime = img.mtime; // for sorting reference

    // Check selection state
    if (sampleState.selectedPaths.has(img.path)) {
        card.classList.add('selected');
    }

    card.innerHTML = `
        <img src="${img.path}" alt="${escapeHTML(img.name)}" loading="lazy" draggable="false">
        <div class="sample-name">${escapeHTML(img.name)}</div>
        <button class="btn-delete-card" title="Delete Image">✕</button>
    `;

    // Delete Card Logic
    card.querySelector('.btn-delete-card').addEventListener('click', (e) => {
        e.stopPropagation(); // Don't trigger selection/lightbox
        showConfirm('Delete Image', `Delete "${img.name}"?`, () => {
            deleteSamples([img.path]);
        });
    });

    // Click Selection Logic (Selection + Open Lightbox)
    card.addEventListener('click', (e) => handleSampleClick(e, img, card));

    // Drag Events
    card.addEventListener('dragstart', handleDragStart);
    card.addEventListener('dragover', handleDragOver);
    card.addEventListener('drop', handleDrop);
    card.addEventListener('dragenter', (e) => e.preventDefault());

    container.appendChild(card);
}

// ==========================================
//  Sample Interactions
// ==========================================

// ==========================================
//  Box Selection (Rubber Band)
// ==========================================
let boxSelection = {
    isSelecting: false,
    startX: 0,
    startY: 0,
    element: null
};

function initSampleInteractions() {
    // Keyboard Navigation
    document.addEventListener('keydown', handleGlobalKeydown);

    // Batch Delete Button
    const btnDelete = $('btn-delete-selected');
    if (btnDelete) {
        btnDelete.addEventListener('click', () => {
            const count = sampleState.selectedPaths.size;
            if (count > 0) {
                showConfirm('Delete Images', `Delete ${count} selected image(s)?`, () => {
                    deleteSamples(Array.from(sampleState.selectedPaths));
                });
            }
        });
    }

    // Box Selection Listeners (on container)
    const container = $('samples-grid'); // This might be hidden initially? 
    // We can attach to document or a wrapper. 
    // Attaching to 'samples-grid' is safest if it exists.
    if (container) {
        container.addEventListener('mousedown', handleBoxStart);
    }
    document.addEventListener('mousemove', handleBoxMove);
    document.addEventListener('mouseup', handleBoxEnd);
}

function handleBoxStart(e) {
    if (e.target.closest('.sample-card')) return;

    if (e.button !== 0) return;

    boxSelection.isSelecting = true;
    sampleState.isExplicitMultiSelect = true;
    boxSelection.startX = e.pageX;
    boxSelection.startY = e.pageY;

    // Create selection box element
    if (!boxSelection.element) {
        const el = document.createElement('div');
        el.className = 'selection-box';
        document.body.appendChild(el);
        boxSelection.element = el;
    }

    const el = boxSelection.element;
    el.style.left = e.pageX + 'px';
    el.style.top = e.pageY + 'px';
    el.style.width = '0px';
    el.style.height = '0px';
    el.style.display = 'block';

    if (!e.ctrlKey && !e.shiftKey) {
        clearSelection();
    }
}

function handleBoxMove(e) {
    if (!boxSelection.isSelecting) return;
    e.preventDefault(); // Stop text selection

    const currentX = e.pageX;
    const currentY = e.pageY;

    const minX = Math.min(boxSelection.startX, currentX);
    const maxX = Math.max(boxSelection.startX, currentX);
    const minY = Math.min(boxSelection.startY, currentY);
    const maxY = Math.max(boxSelection.startY, currentY);

    const el = boxSelection.element;
    el.style.left = minX + 'px';
    el.style.top = minY + 'px';
    el.style.width = (maxX - minX) + 'px';
    el.style.height = (maxY - minY) + 'px';

    // Update selection in real-time
    updateBoxSelection(minX, minY, maxX, maxY, e.ctrlKey);
}

function handleBoxEnd(e) {
    if (!boxSelection.isSelecting) return;
    boxSelection.isSelecting = false;
    if (boxSelection.element) {
        boxSelection.element.style.display = 'none';
    }
}

function updateBoxSelection(x1, y1, x2, y2, isCtrl) {
    const cards = document.querySelectorAll('.sample-card');

    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        // Get card coordinates relative to page (since box uses pageX/Y)
        const cardX1 = rect.left + window.scrollX;
        const cardY1 = rect.top + window.scrollY;
        const cardX2 = cardX1 + rect.width;
        const cardY2 = cardY1 + rect.height;

        // Check intersection
        const isOverlapping = !(cardX1 > x2 || cardX2 < x1 || cardY1 > y2 || cardY2 < y1);

        if (isOverlapping) {
            sampleState.selectedPaths.add(card.dataset.path);
        } else if (!isCtrl) {
            // If not holding Ctrl, box selection is "set" logic, but real-time clearing 
            // of things outside box is tricky if we started with a selection.
            // Simplified: Box Selection ADDS to selection during drag.
            // If we want "Select ONLY these", we cleared at start.
            // Scaling back: Standard behavior is Additive if Box touches.

            // To be strict: 
            // If we cleared at start, then sampleState contains only what is currently overlapping.
            // But we need to NOT delete things we just added in this drag session if we shrink box.
            // This requires "initialSelection" state. Too complex for raw JS in one function.
            // CURRENT LOGIC: additive only during move. 
            // If user shrinks box, items stay selected. (Minor UX quirk but acceptable).
        }
    });
    updateSelectionVisuals();
}

function handleSampleClick(e, img, card) {
    // Lightbox triggers on double click or specific action? 
    // User request: "arrow keys to move... even if user is open a specific image"
    // Standard UI: Click = Select, Double Click = Open? 
    // Or Click = Open? 
    // Plan: Click = Select. Double Click = Lightbox.
    // If modifier keys are used, strictly selection.

    // BUT user said "open a specific image", implying lightbox.
    // Let's implement: Click selects. Double click opens.
    // Also, if you just click and no modifiers, maybe open? 
    // "select multiple images then they can drag" -> implies single click might select.

    // Hybrid approach:
    // Simple Click: Selects (and clears others)
    // Ctrl+Click: Toggles
    // Shift+Click: Range
    // Double Click: Open Lightbox

    if (e.ctrlKey || e.metaKey) {
        sampleState.isExplicitMultiSelect = true;
        toggleSelection(img.path);
    } else if (e.shiftKey) {
        sampleState.isExplicitMultiSelect = true;
        selectRange(img.path);
    } else {
        // Simple click: Select and Open Lightbox
        sampleState.isExplicitMultiSelect = false;
        selectSingle(img.path);
        openLightbox(img.path, img.name);
    }
    sampleState.lastSelectedPath = img.path;
}

// Better to attach dblclick to card in createSampleCard, adding it here implies logic change
// Lets add logic in createSampleCard wrapper
// (Modified createSampleCard above needs dblclick listener)

function selectSingle(path) {
    sampleState.selectedPaths.clear();
    sampleState.selectedPaths.add(path);
    updateSelectionVisuals();
}

function toggleSelection(path) {
    if (sampleState.selectedPaths.has(path)) {
        sampleState.selectedPaths.delete(path);
    } else {
        sampleState.selectedPaths.add(path);
    }
    updateSelectionVisuals();
}

function selectRange(targetPath) {
    if (!sampleState.lastSelectedPath) {
        selectSingle(targetPath);
        return;
    }

    // Find indices in the flattened visual list
    // To do this right, we need the current visual DOM order
    const allCards = Array.from(document.querySelectorAll('.sample-card'));
    const startIdx = allCards.findIndex(c => c.dataset.path === sampleState.lastSelectedPath);
    const endIdx = allCards.findIndex(c => c.dataset.path === targetPath);

    if (startIdx === -1 || endIdx === -1) return;

    const [min, max] = [Math.min(startIdx, endIdx), Math.max(startIdx, endIdx)];

    // Add range
    // If ctrl not held, clear others? Standard behavior is usually yes for Shift-click
    // But lets keep it additive for now or clear? 
    // Windows Explorer: Shift-click clears previous selection (except anchor)
    // Let's clear for simplicity.
    sampleState.selectedPaths.clear();
    for (let i = min; i <= max; i++) {
        sampleState.selectedPaths.add(allCards[i].dataset.path);
    }
    updateSelectionVisuals();
}

function updateSelectionVisuals() {
    const isMultiRoot = sampleState.selectedPaths.size > 1 || sampleState.isExplicitMultiSelect;
    const count = sampleState.selectedPaths.size;

    // Batch delete button
    const btnDelete = $('btn-delete-selected');
    if (btnDelete) {
        if (count > 0) {
            btnDelete.classList.remove('hidden');
            btnDelete.textContent = `🗑️ Delete (${count})`;
        } else {
            btnDelete.classList.add('hidden');
        }
    }

    // Toggle multi-select mode on all grids
    document.querySelectorAll('.group-grid').forEach(grid => {
        grid.classList.toggle('multi-select-mode', isMultiRoot);
    });

    document.querySelectorAll('.sample-card').forEach(card => {
        if (sampleState.selectedPaths.has(card.dataset.path)) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}

function clearSelection() {
    sampleState.selectedPaths.clear();
    updateSelectionVisuals();
}

// Drag and Drop Logic
function handleDragStart(e) {
    const path = e.target.closest('.sample-card').dataset.path;

    // If dragging an unselected item, select it first
    if (!sampleState.selectedPaths.has(path)) {
        selectSingle(path);
    }

    e.dataTransfer.setData('text/plain', JSON.stringify(Array.from(sampleState.selectedPaths)));
    e.dataTransfer.effectAllowed = 'move';
    e.target.closest('.sample-card').classList.add('dragging');
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';

    // Remove existing drag indicators
    document.querySelectorAll('.drag-over-left, .drag-over-right, .drag-over-grid').forEach(el => {
        el.classList.remove('drag-over-left', 'drag-over-right', 'drag-over-grid');
    });

    const targetCard = e.target.closest('.sample-card');
    const targetGrid = e.target.closest('.group-grid');

    if (targetCard) {
        const rect = targetCard.getBoundingClientRect();
        const relX = e.clientX - rect.left;
        if (relX < rect.width / 2) {
            targetCard.classList.add('drag-over-left');
        } else {
            targetCard.classList.add('drag-over-right');
        }
    } else if (targetGrid) {
        // Visual feedback for dropping into the grid background
        targetGrid.classList.add('drag-over-grid');
    }
}

function handleDrop(e) {
    e.preventDefault();
    document.querySelectorAll('.drag-over-left, .drag-over-right, .drag-over-grid').forEach(el => {
        el.classList.remove('drag-over-left', 'drag-over-right', 'drag-over-grid');
    });

    const targetCard = e.target.closest('.sample-card');
    const targetGrid = e.target.closest('.group-grid');

    if (!targetGrid) return;

    try {
        const paths = JSON.parse(e.dataTransfer.getData('text/plain'));
        const allCards = Array.from(document.querySelectorAll('.sample-card'));
        const cardsToMove = allCards.filter(c => paths.includes(c.dataset.path));

        if (targetCard) {
            const rect = targetCard.getBoundingClientRect();
            const relX = e.clientX - rect.left;
            const insertBefore = relX < rect.width / 2;

            cardsToMove.forEach(card => {
                if (insertBefore) {
                    targetGrid.insertBefore(card, targetCard);
                } else {
                    targetGrid.insertBefore(card, targetCard.nextSibling);
                }
            });
        } else {
            // Drop in grid background -> Append to end
            cardsToMove.forEach(card => {
                targetGrid.appendChild(card);
            });
        }

        cardsToMove.forEach(card => card.classList.remove('dragging'));

        // Save the new state permanently
        saveManualOrder();

    } catch (err) {
        console.error("Drop error", err);
    }
}

// Keyboard Navigation & Lightbox
function handleGlobalKeydown(e) {
    // Lightbox navigation
    const lightbox = document.querySelector('.lightbox');
    if (lightbox) {
        const currentSrc = lightbox.querySelector('img').getAttribute('src');
        handleLightboxNavigation(e, currentSrc, lightbox);
        return;
    }

    // Grid navigation (Arrow Keys)
    // Only if focus is not in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key.startsWith('Arrow')) {
        e.preventDefault();
        navigateGrid(e.key, e.ctrlKey);
    }

    if (e.key === 'Enter') {
        const selected = Array.from(sampleState.selectedPaths);
        if (selected.length === 1) {
            const card = document.querySelector(`.sample-card[data-path="${selected[0]}"]`);
            openLightbox(selected[0], card ? card.dataset.name : '');
        }
    }
}

// Navigation Helper
function calculateNextIndex(allCards, currentIdx, direction) {
    if (currentIdx === -1) return 0;

    let nextIdx = currentIdx;

    if (direction === 'ArrowRight') nextIdx = Math.min(currentIdx + 1, allCards.length - 1);
    if (direction === 'ArrowLeft') nextIdx = Math.max(currentIdx - 1, 0);

    if (direction === 'ArrowUp' || direction === 'ArrowDown') {
        const currentRect = allCards[currentIdx].getBoundingClientRect();
        const currentCenter = currentRect.left + currentRect.width / 2;
        const currentY = currentRect.top + currentRect.height / 2;

        let bestDist = Infinity;
        let bestCandidate = -1;

        allCards.forEach((c, i) => {
            if (i === currentIdx) return;
            const r = c.getBoundingClientRect();
            const y = r.top + r.height / 2;
            const x = r.left + r.width / 2;

            // Metric: Minimize vertical dist first, then horizontal.
            const distV = Math.abs(y - currentY);
            const distH = Math.abs(x - currentCenter);
            const score = distV * 2 + distH;

            if ((direction === 'ArrowUp' && y < currentRect.top) ||
                (direction === 'ArrowDown' && y > currentRect.bottom)) {
                if (score < bestDist) {
                    bestDist = score;
                    bestCandidate = i;
                }
            }
        });

        if (bestCandidate !== -1) nextIdx = bestCandidate;
    }
    return nextIdx;
}

function navigateGrid(direction, isCtrl) {
    // Find current focus (last selected)
    // If no selection, select first
    const allCards = Array.from(document.querySelectorAll('.sample-card'));
    if (allCards.length === 0) return;

    let idx = -1;
    if (sampleState.lastSelectedPath) {
        idx = allCards.findIndex(c => c.dataset.path === sampleState.lastSelectedPath);
    }

    if (idx === -1) {
        selectSingle(allCards[0].dataset.path);
        allCards[0].scrollIntoView({ block: 'center' });
        sampleState.lastSelectedPath = allCards[0].dataset.path;
        return;
    }

    const nextIdx = calculateNextIndex(allCards, idx, direction);

    if (nextIdx !== idx) {
        const path = allCards[nextIdx].dataset.path;
        if (!isCtrl) {
            selectSingle(path);
        } else {
            selectSingle(path);
        }
        allCards[nextIdx].scrollIntoView({ block: 'nearest' });
        sampleState.lastSelectedPath = path; // Update visual focus anchor
    }
}

function openLightbox(src, name) {
    // Remove existing
    const existing = document.querySelector('.lightbox');
    if (existing) existing.remove();

    const lb = document.createElement('div');
    lb.className = 'lightbox';
    lb.innerHTML = `
        <div class="lightbox-title">${name || ''}</div>
        <img src="${src}">
        <div class="lightbox-metadata hidden"></div>
        <div class="lightbox-nav">
            Use Arrow Keys to navigate | ESC to close
        </div>
    `;

    // Click background to close
    lb.addEventListener('click', (e) => {
        if (e.target === lb) lb.remove();
    });

    document.body.appendChild(lb);
    loadLightboxMetadata(src, lb);

    // Auto-fade navigation hint after 3s
    setTimeout(() => {
        const nav = lb.querySelector('.lightbox-nav');
        if (nav) nav.style.opacity = '0';
    }, 3000);
}

function handleLightboxNavigation(e, currentSrc, lightbox) {
    if (e.key === 'Escape') {
        lightbox.remove();
        return;
    }

    if (!e.key.startsWith('Arrow')) return;

    e.preventDefault();

    // Find current index
    const allCards = Array.from(document.querySelectorAll('.sample-card'));
    const idx = allCards.findIndex(c => c.dataset.path === currentSrc);
    if (idx === -1) return;

    const nextIdx = calculateNextIndex(allCards, idx, e.key);

    if (nextIdx !== idx) {
        const nextCard = allCards[nextIdx];
        const nextPath = nextCard.dataset.path;
        const nextName = nextCard.dataset.name;

        lightbox.querySelector('img').src = nextPath;
        const titleEl = lightbox.querySelector('.lightbox-title');
        if (titleEl) titleEl.textContent = nextName || '';

        // Update metadata
        loadLightboxMetadata(nextPath, lightbox);

        // Also update selection in background
        selectSingle(nextPath);
        nextCard.scrollIntoView({ block: 'nearest' });
    }
}

async function loadLightboxMetadata(path, lightbox) {
    const metaEl = lightbox.querySelector('.lightbox-metadata');
    if (!metaEl) return;

    // Convert path from /api/jobs/NAME/samples/... to /api/jobs/NAME/metadata/...
    const metaUrl = path.replace('/samples/', '/metadata/');

    try {
        const res = await fetch(metaUrl);
        if (!res.ok) throw new Error();
        const data = await res.json();

        if (data.parameters) {
            metaEl.textContent = data.parameters;
            metaEl.classList.remove('hidden');
        } else {
            metaEl.classList.add('hidden');
        }
    } catch (e) {
        metaEl.classList.add('hidden');
    }
}

// Add double click listener helper
function addDoubleClick(element, callback) {
    let lastClick = 0;
    element.addEventListener('click', (e) => {
        const now = new Date().getTime();
        if (now - lastClick < 300) {
            callback(e);
        }
        lastClick = now;
    });
}


// ==========================================
//  TensorBoard
// ==========================================

let tbUrl = null;

async function checkTensorBoard() {
    if (!currentJob) return;
    const status = await api(`/api/jobs/${currentJob}/tensorboard/status`);
    updateTbState(status.running, status.url);
}

function updateTbState(running, url) {
    $('btn-tb-launch').classList.toggle('hidden', running);
    $('btn-tb-stop').classList.toggle('hidden', !running);
    $('btn-tb-open').classList.toggle('hidden', !running);
    $('tb-status').textContent = running ? `Running on port ${new URL(url).port}` : 'Not running';
    $('tb-status').style.color = running ? 'var(--success)' : 'var(--text-muted)';

    if (running && url) {
        tbUrl = url;
        $('tb-placeholder').classList.add('hidden');
        $('tb-iframe').classList.remove('hidden');
        // Only set src if it changed
        if ($('tb-iframe').src !== url) {
            $('tb-iframe').src = url;
        }
    } else {
        tbUrl = null;
        $('tb-placeholder').classList.remove('hidden');
        $('tb-iframe').classList.add('hidden');
        $('tb-iframe').src = '';
    }
}

async function launchTensorBoard() {
    if (!currentJob) return;
    $('btn-tb-launch').disabled = true;
    $('btn-tb-launch').textContent = 'Starting...';

    const result = await api(`/api/jobs/${currentJob}/tensorboard`, { method: 'POST' });

    if (result.error) {
        alert(result.error);
        $('btn-tb-launch').disabled = false;
        $('btn-tb-launch').textContent = '\uD83D\uDE80 Launch';
        return;
    }

    // Give TensorBoard a moment to start
    setTimeout(() => {
        updateTbState(true, result.url);
        $('btn-tb-launch').disabled = false;
        $('btn-tb-launch').textContent = '\uD83D\uDE80 Launch';
        showToast('TensorBoard launched');
    }, 2000);
}

async function stopTensorBoard() {
    if (!currentJob) return;
    await api(`/api/jobs/${currentJob}/tensorboard/stop`, { method: 'POST' });
    updateTbState(false, null);
    showToast('TensorBoard stopped');
}

// ==========================================
//  Global Settings
// ==========================================

function applyTheme(theme) {
    const t = theme || 'github-dark';
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('ui_theme', t);
}

async function loadGlobalSettings() {
    const config = await api('/api/global-config');
    $('cfg-global-dit').value = config.model_paths?.dit_path || '';
    $('cfg-global-qwen3').value = config.model_paths?.qwen3_path || '';
    $('cfg-global-vae').value = config.model_paths?.vae_path || '';
    $('cfg-global-venv').value = config.venv_path || '';

    // Theme
    const theme = config.ui?.theme || 'github-dark';
    $('cfg-theme').value = theme;
    applyTheme(theme);

    // Background settings
    const pos = config.ui?.background_position || '50% 50%';
    const dim = config.ui?.dim_level ?? 70;
    const brightness = config.ui?.brightness_level ?? 100;
    const blur = config.ui?.blur_level ?? 10;
    const textShadow = config.ui?.text_shadow_size ?? 0;

    $('cfg-bg-dim').value = dim;
    $('val-bg-dim').textContent = dim + '%';
    $('cfg-bg-brightness').value = brightness;
    $('val-bg-brightness').textContent = brightness + '%';
    $('cfg-bg-blur').value = blur;
    $('val-bg-blur').textContent = blur + 'px';
    $('cfg-text-shadow').value = textShadow;
    $('val-text-shadow').textContent = textShadow + 'px';

    if (config.ui?.background) {
        applyBackground(config.ui.background, pos, dim, brightness, blur, textShadow);
        $('bg-visual-controls').classList.remove('hidden');
    } else {
        $('bg-pos-group').classList.add('hidden');
        $('bg-visual-controls').classList.add('hidden');
    }
}

async function saveGlobalSettings() {
    // Read existing config first to preserve bg settings
    const existingConfig = await api('/api/global-config');

    const config = {
        model_paths: {
            dit_path: $('cfg-global-dit').value,
            qwen3_path: $('cfg-global-qwen3').value,
            vae_path: $('cfg-global-vae').value
        },
        venv_path: $('cfg-global-venv').value,
        ui: {
            ...(existingConfig.ui || {}),
            theme: $('cfg-theme').value,
            background_position: `${bgPosPercent.x.toFixed(1)}% ${bgPosPercent.y.toFixed(1)}%`,
            dim_level: parseInt($('cfg-bg-dim').value),
            brightness_level: parseInt($('cfg-bg-brightness').value),
            blur_level: parseInt($('cfg-bg-blur').value),
            text_shadow_size: parseInt($('cfg-text-shadow').value)
        }
    };

    // Apply theme immediately
    applyTheme(config.ui.theme);

    // Live update background if one exists
    if (existingConfig?.ui?.background) {
        applyBackground(
            existingConfig.ui.background,
            config.ui.background_position,
            config.ui.dim_level,
            config.ui.brightness_level,
            config.ui.blur_level,
            config.ui.text_shadow_size
        );
    }

    await api('/api/global-config', { method: 'PUT', body: config });
    closeModal('modal-global-settings');
    showToast('Global settings saved');
}

// === Background Image Functions ===

function applyBackground(url, position = '50% 50%', dim = 70, brightness = 100, blur = 10, textShadow = 0) {
    const appContainer = document.querySelector('.app');
    const preview = $('bg-drag-preview');
    const handle = $('bg-drag-handle');

    // Cache for early load
    localStorage.setItem('ui_background', JSON.stringify({ url, position, dim, brightness, blur, textShadow }));

    // Remove the early-load style once we have the real container
    const earlyStyle = document.getElementById('early-bg');
    if (earlyStyle) earlyStyle.remove();

    if (url && url !== 'none' && url !== '') {
        const root = document.documentElement;
        appContainer.style.backgroundImage = `url('${url}')`;
        appContainer.style.backgroundPosition = position;
        appContainer.classList.add('has-bg');
        root.style.setProperty('--bg-dim', dim / 100);
        root.style.setProperty('--bg-brightness', brightness / 100);
        root.style.setProperty('--bg-blur', blur + 'px');
        root.style.setProperty('--text-shadow-size', textShadow + 'px');

        preview.style.backgroundImage = `url('${url}')`;
        preview.style.backgroundPosition = position;

        const parts = position.split(' ');
        if (parts.length === 2) {
            bgPosPercent.x = parseFloat(parts[0]);
            bgPosPercent.y = parseFloat(parts[1]);
            handle.style.left = bgPosPercent.x + '%';
            handle.style.top = bgPosPercent.y + '%';
        }

        $('btn-remove-bg').classList.remove('hidden');
        $('bg-pos-group').classList.remove('hidden');
        $('bg-visual-controls').classList.remove('hidden');
    } else {
        appContainer.style.backgroundImage = 'none';
        appContainer.classList.remove('has-bg');
        $('btn-remove-bg').classList.add('hidden');
        $('bg-pos-group').classList.add('hidden');
        $('bg-visual-controls').classList.add('hidden');
    }
}

// Drag logic
function updateBgPosFromMouse(e) {
    const container = $('bg-drag-container');
    const rect = container.getBoundingClientRect();
    let x = ((e.clientX - rect.left) / rect.width) * 100;
    let y = ((e.clientY - rect.top) / rect.height) * 100;
    x = Math.max(0, Math.min(100, x));
    y = Math.max(0, Math.min(100, y));
    bgPosPercent = { x, y };
    const posStr = `${x.toFixed(1)}% ${y.toFixed(1)}%`;
    $('bg-drag-handle').style.left = x + '%';
    $('bg-drag-handle').style.top = y + '%';
    $('bg-drag-preview').style.backgroundPosition = posStr;
    document.querySelector('.app').style.backgroundPosition = posStr;
}

$('bg-drag-container').onmousedown = (e) => {
    isDraggingBg = true;
    updateBgPosFromMouse(e);
};
window.addEventListener('mousemove', (e) => {
    if (isDraggingBg) updateBgPosFromMouse(e);
});
window.addEventListener('mouseup', () => { isDraggingBg = false; });

// Upload handler
$('cfg-bg-upload').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
        const base64 = event.target.result;
        const res = await api('/api/global/background', {
            method: 'POST',
            body: { image: base64 }
        });
        if (res.success) {
            const config = await api('/api/global-config');
            const pos = `${bgPosPercent.x}% ${bgPosPercent.y}%`;
            const dim = parseInt($('cfg-bg-dim').value);
            const brightness = parseInt($('cfg-bg-brightness').value);
            const blur = parseInt($('cfg-bg-blur').value);
            const textShadow = parseInt($('cfg-text-shadow').value);
            applyBackground(res.url, pos, dim, brightness, blur, textShadow);
            // Save to global config
            config.ui = config.ui || {};
            config.ui.background = res.url;
            config.ui.background_position = pos;
            config.ui.dim_level = dim;
            config.ui.brightness_level = brightness;
            config.ui.blur_level = blur;
            config.ui.text_shadow_size = textShadow;
            await api('/api/global-config', { method: 'PUT', body: config });
            showToast('Background updated!');
        }
    };
    reader.readAsDataURL(file);
};

// Remove handler
$('btn-remove-bg').onclick = async () => {
    await api('/api/global/background', { method: 'DELETE' });
    applyBackground(null);
    const config = await api('/api/global-config');
    if (config.ui) delete config.ui.background;
    await api('/api/global-config', { method: 'PUT', body: config });
    showToast('Background removed');
};

// Slider live previews
$('cfg-bg-dim').oninput = (e) => {
    $('val-bg-dim').textContent = e.target.value + '%';
    document.documentElement.style.setProperty('--bg-dim', e.target.value / 100);
};
$('cfg-bg-brightness').oninput = (e) => {
    $('val-bg-brightness').textContent = e.target.value + '%';
    document.documentElement.style.setProperty('--bg-brightness', e.target.value / 100);
};
$('cfg-bg-blur').oninput = (e) => {
    $('val-bg-blur').textContent = e.target.value + 'px';
    document.documentElement.style.setProperty('--bg-blur', e.target.value + 'px');
};
$('cfg-text-shadow').oninput = (e) => {
    $('val-text-shadow').textContent = e.target.value + 'px';
    document.documentElement.style.setProperty('--text-shadow-size', e.target.value + 'px');
};

// Theme change handler (live preview)
$('cfg-theme').onchange = (e) => {
    applyTheme(e.target.value);
};

// ==========================================
//  Modals & Helpers
// ==========================================

function openModal(id) {
    $(id).classList.remove('hidden');
}

function closeModal(id) {
    $(id).classList.add('hidden');
}

function showConfirm(title, message, onConfirm) {
    $('confirm-title').textContent = title;
    $('confirm-message').textContent = message;
    const actions = $('confirm-actions');
    actions.innerHTML = '';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn btn-ghost';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.onclick = () => closeModal('modal-confirm');

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'btn btn-danger';
    confirmBtn.textContent = 'Confirm';
    confirmBtn.onclick = () => {
        closeModal('modal-confirm');
        onConfirm();
    };

    actions.appendChild(cancelBtn);
    actions.appendChild(confirmBtn);
    openModal('modal-confirm');
}

function showToast(msg) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; z-index: 300;
        padding: 12px 20px; border-radius: 8px;
        background: var(--bg-tertiary); border: 1px solid var(--border);
        color: var(--text-primary); font-size: 0.9rem;
        box-shadow: var(--shadow); animation: fadeIn 0.2s;
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ==========================================
//  Tabs
// ==========================================

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => {
            p.classList.remove('active');
            p.classList.add('hidden');
        });
        tab.classList.add('active');
        const pane = $(`tab-${tab.dataset.tab}`);
        pane.classList.remove('hidden');
        pane.classList.add('active');

        localStorage.setItem('lastTab', tab.dataset.tab);

        // Stop polling if switching away from samples (or just reset it)
        if (samplesPollTimer) {
            clearInterval(samplesPollTimer);
            samplesPollTimer = null;
        }

        // Auto-refresh data on tab switch
        if (tab.dataset.tab === 'samples') {
            loadSamples();
            samplesPollTimer = setInterval(() => loadSamples(true), 3000);
        }
        if (tab.dataset.tab === 'prompts') loadPrompts();
        if (tab.dataset.tab === 'tensorboard') checkTensorBoard();
    });
});

// ==========================================
//  Event Listeners
// ==========================================

$('cfg-enable-sampling').addEventListener('change', (e) => {
    $('group-sample-every').classList.toggle('hidden', !e.target.checked);
});

document.querySelectorAll('input[name="duration-unit"]').forEach(el => {
    el.addEventListener('change', updateDurationUnit);
});

function updateDurationUnit() {
    const unit = document.querySelector('input[name="duration-unit"]:checked').value;
    const isEpochs = unit === 'epochs';

    $('schedule-epochs').classList.toggle('hidden', !isEpochs);
    $('schedule-steps').classList.toggle('hidden', isEpochs);

    $('container-sample-every-epochs').classList.toggle('hidden', !isEpochs);
    $('container-sample-every-steps').classList.toggle('hidden', isEpochs);
}

// New Job
$('btn-new-job').addEventListener('click', () => {
    $('new-job-name').value = 'my_job';
    openModal('modal-new-job');
    $('new-job-name').focus();
});

$('btn-create-job').addEventListener('click', async () => {
    const name = $('new-job-name').value.trim();
    if (!name) return;
    const result = await api('/api/jobs', { method: 'POST', body: { name } });
    if (result.error) {
        alert(result.error);
        return;
    }
    closeModal('modal-new-job');
    await loadJobs();
    selectJob(result.name);
    showToast('Job created');
});

// Load GPUs from server
async function loadGPUs() {
    const container = $('cfg-gpu-selection');
    try {
        const gpus = await api('/api/system/gpus');
        container.innerHTML = '';

        if (gpus.length === 0) {
            container.innerHTML = '<small>No NVIDIA GPUs detected (CPU only).</small>';
            return;
        }

        gpus.forEach(gpu => {
            const card = document.createElement('div');
            card.className = 'gpu-card selected'; // Default to all selected
            card.dataset.index = gpu.index;
            card.id = `gpu-card-${gpu.index}`;

            card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <div class="gpu-status">
                    <div class="status-dot"></div>
                    <span class="gpu-status-text">Idle</span>
                </div>
                <input type="checkbox" name="gpu-select" value="${gpu.index}" checked id="gpu-${gpu.index}">
            `;

            card.addEventListener('click', (e) => {
                if (e.target.tagName === 'INPUT') {
                    card.classList.toggle('selected', e.target.checked);
                    return;
                }
                cb.checked = !cb.checked;
                card.classList.toggle('selected', cb.checked);
                checkDirty();
            });


            container.appendChild(card);
        });

        updateGPUActivity();
    } catch (err) {
        console.error("Failed to load GPUs:", err);
        container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
    }
}

// Load GPUs for generation (separate from training GPU selection)
async function loadGenGPUs() {
    const container = $('gen-gpu-selection');
    if (!container) return;
    try {
        const gpus = await api('/api/system/gpus');
        container.innerHTML = '';

        if (gpus.length === 0) {
            container.innerHTML = '<small>No NVIDIA GPUs detected.</small>';
            return;
        }

        gpus.forEach((gpu, i) => {
            const card = document.createElement('div');
            card.className = 'gpu-card' + (i === 0 ? ' selected' : '');
            card.dataset.index = gpu.index;
            card.id = `gen-gpu-card-${gpu.index}`;

            card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <input type="checkbox" name="gen-gpu-select" value="${gpu.index}" ${i === 0 ? 'checked' : ''} id="gen-gpu-${gpu.index}">
            `;

            const cb = card.querySelector('input[type=checkbox]');
            card.addEventListener('click', (e) => {
                if (e.target.tagName === 'INPUT') {
                    card.classList.toggle('selected', e.target.checked);
                    updateGenGPULabel();
                    return;
                }
                cb.checked = !cb.checked;
                card.classList.toggle('selected', cb.checked);
                updateGenGPULabel();
            });

            container.appendChild(card);
        });

        updateGenGPULabel();
    } catch (err) {
        console.error('Failed to load gen GPUs:', err);
        container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
    }
}

function getSelectedGenGPUs() {
    const checked = document.querySelectorAll('input[name="gen-gpu-select"]:checked');
    return Array.from(checked).map(c => c.value).join(',');
}

function restoreGenGPUSelection(gpuIds) {
    if (!gpuIds) return;
    const ids = gpuIds.split(',').map(s => s.trim());
    document.querySelectorAll('input[name="gen-gpu-select"]').forEach(cb => {
        cb.checked = ids.includes(cb.value);
        const card = cb.closest('.gpu-card');
        if (card) card.classList.toggle('selected', cb.checked);
    });
    updateGenGPULabel();
}

function updateGenGPULabel() {
    const label = $('gen-gpu-mode-label');
    const optionsDiv = $('gen-multi-gpu-options');
    if (!label) return;
    const selected = document.querySelectorAll('input[name="gen-gpu-select"]:checked');
    if (selected.length > 1) {
        label.textContent = '— Multi-GPU';
        label.style.color = 'var(--success)';
        if (optionsDiv) optionsDiv.style.display = 'block';
    } else {
        label.textContent = '';
        label.style.color = '';
        if (optionsDiv) optionsDiv.style.display = 'none';
    }
}

async function updateGPUActivity() {
    try {
        const res = await fetch('/api/gpu/activity');
        if (!res.ok) return;
        const activity = await res.json(); // { "0": "training", "1": "sampling" }

        document.querySelectorAll('.gpu-card').forEach(card => {
            const index = card.dataset.index;
            const status = activity[index] || 'idle';
            const textEl = card.querySelector('.gpu-status-text');

            card.classList.remove('active-training', 'active-sampling');

            if (status === 'training') {
                card.classList.add('active-training');
                textEl.textContent = 'Training';
            } else if (status === 'sampling') {
                card.classList.add('active-sampling');
                textEl.textContent = 'Sampling';
            } else {
                textEl.textContent = 'Idle';
            }
        });
    } catch (err) {
        // Silently fail polling
    }
}

$('btn-cancel-job').addEventListener('click', () => closeModal('modal-new-job'));

// Enter key in new job name
$('new-job-name').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('btn-create-job').click();
});

// Save
$('btn-save').addEventListener('click', saveJob);

// Keyboard shortcut: Ctrl+S
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        if (currentJob && isDirty) saveJob();
    }
});

// Clone
$('btn-clone').addEventListener('click', () => {
    if (!currentJob) return;

    // Calculate default name
    let defaultName = `${currentJob}_copy`;
    let counter = 1;
    const uniqueName = (base) => {
        const jobItems = document.querySelectorAll('.job-name');
        for (let item of jobItems) {
            if (item.textContent === base) return false;
        }
        return true;
    };

    while (!uniqueName(defaultName)) {
        counter++;
        defaultName = `${currentJob}_copy_${counter}`;
    }

    // Open Modal
    $('clone-job-name').value = defaultName;
    openModal('modal-clone-job');
    $('clone-job-name').focus();
    $('clone-job-name').select();
});

// Confirm Clone
$('btn-confirm-clone').addEventListener('click', async () => {
    const newName = $('clone-job-name').value.trim();
    if (!newName) return;

    const result = await api(`/api/jobs/${currentJob}/clone`, {
        method: 'POST',
        body: { newName: newName }
    });

    if (result.error) {
        alert(result.error);
        return;
    }
    closeModal('modal-clone-job');
    await loadJobs();
    selectJob(result.name);
    showToast('Job cloned');
});

// Cancel Clone
$('btn-cancel-clone').addEventListener('click', () => closeModal('modal-clone-job'));

// Enter key in clone job name
$('clone-job-name').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('btn-confirm-clone').click();
});

// Delete
$('btn-delete').addEventListener('click', () => {
    if (!currentJob) return;
    showConfirm('Delete Job', `Delete "${currentJob}" and all its files? This cannot be undone.`, async () => {
        const deletedJob = currentJob;
        await api(`/api/jobs/${deletedJob}`, { method: 'DELETE' });

        // Clean up all localStorage keys for the deleted job
        localStorage.removeItem(`prompt_transient_${deletedJob}`);
        localStorage.removeItem(`sample_order_${deletedJob}`);
        localStorage.removeItem('lastJob');

        currentJob = null;
        isDirty = false;
        $('btn-save').classList.add('hidden');
        $('btn-discard').classList.add('hidden');
        emptyState.classList.remove('hidden');

        jobEditor.classList.add('hidden');
        await loadJobs();
        showToast('Job deleted');
    });
});

// Train
$('btn-run').addEventListener('click', async () => {
    if (!currentJob) return;

    let warningMsg = '';
    // Check sampling Logic
    if ($('cfg-enable-sampling').checked && currentPrompts.length === 0) {
        warningMsg = "Sampling is enabled but no prompts are defined.\n\nContinue training without generating samples...\n\n";
    }

    // Auto-save first
    if (isDirty) await saveJob();

    const result = await api(`/api/jobs/${currentJob}/train/start`, { method: 'POST' });
    if (result.error) {
        alert(result.error);
        return;
    }
    updateRunningState(true);
    consoleOutput.textContent = '';
    if (warningMsg) appendConsole(warningMsg);

    // Auto-switch to console tab
    document.querySelector('[data-tab="console"]').click();
    showToast('Training started');
});

// Generate
$('btn-gen-sample').addEventListener('click', async () => {
    if (!currentJob) return;
    savePromptTransientSettings();
    if (isDirty) await saveJob();

    if (currentPrompts.length === 0) {
        showToast('Add sample prompts first');
        return;
    }

    const payload = {};
    const loraPath = $('gen-lora-select').value;
    if (loraPath) {
        payload.network_weights = loraPath;
        payload.network_mul = parseFloat($('gen-lora-mul').value) || 1.0;
    }
    // Add Anima generation params
    payload.flow_shift = parseFloat($('cfg-flow-shift').value) || 3.0;
    payload.flash_attn = $('gen-flash-attn').checked;
    payload.sage_attn = $('gen-sage-attn').checked;
    payload.gen_gpu_ids = getSelectedGenGPUs();
    payload.gen_multi_gpu_mode = $('gen-multi-gpu-mode').value;

    const result = await api(`/api/jobs/${currentJob}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: Object.assign(payload, {
            keep_loaded: $('chk-keep-loaded').checked
        })
    });

    if (result.error) {
        alert(result.error);
        return;
    }
    // updateRunningState(true); // Don't block 'Train' button for generation
    // consoleOutput.textContent = ...; // Don't wipe console!
    appendConsole(`Starting generation...\n${loraPath ? `Using LoRA: ${loraPath} (x${payload.network_mul})` : '(Using base model)'}\nFlow Shift: ${payload.flow_shift}\n\n`);

    // Auto-switch to console tab -> REMOVED per user request
    // document.querySelector('[data-tab="console"]').click();
    showToast('Generation started');
});

// Unload Model
$('btn-unload-model').addEventListener('click', async () => {
    if (!currentJob) return;
    showToast('Unloading model...');
    const result = await api(`/api/jobs/${currentJob}/unload`, { method: 'POST' });
    if (result.success) {
        showToast(result.message || 'Model unloaded');
    } else {
        alert(result.error);
    }
});

$('btn-refresh-checkpoints').addEventListener('click', () => {
    loadCheckpoints();
    showToast('Checkpoints refreshed');
});

// Stop
$('btn-stop').addEventListener('click', () => {
    if (!currentJob) return;
    showConfirm('Stop Training', `Stop training for "${currentJob}"?`, async () => {
        await api(`/api/jobs/${currentJob}/train/stop`, { method: 'POST' });
        updateRunningState(false);
        showToast('Training stopped');
    });
});

// Console clear
$('btn-clear-console').addEventListener('click', () => {
    consoleOutput.textContent = 'Waiting for training to start...';
});

// Samples refresh
$('btn-refresh-samples').addEventListener('click', loadSamples);

// TensorBoard
$('btn-tb-launch').addEventListener('click', launchTensorBoard);
$('btn-tb-stop').addEventListener('click', () => {
    showConfirm('Stop TensorBoard', 'Stop the TensorBoard server for this job?', stopTensorBoard);
});
$('btn-tb-open').addEventListener('click', () => {
    if (tbUrl) window.open(tbUrl, '_blank');
});

// Global Settings
$('btn-global-settings').addEventListener('click', () => {
    loadGlobalSettings();
    openModal('modal-global-settings');
});

$('btn-close-global').addEventListener('click', () => closeModal('modal-global-settings'));
$('btn-save-global').addEventListener('click', saveGlobalSettings);

// Prompts
$('btn-add-prompt').addEventListener('click', addPrompt);
$('btn-apply-global').addEventListener('click', applyGlobalSettings);

// Persistence for Prompt Tab settings
['gen-lora-select', 'gen-lora-mul', 'chk-keep-loaded', 'gen-flash-attn', 'gen-multi-gpu-mode', 'global-w', 'global-h', 'global-s', 'global-l', 'global-d'].forEach(id => {
    $(id).addEventListener('change', savePromptTransientSettings);
    if ($(id).tagName === 'INPUT') {
        $(id).addEventListener('input', savePromptTransientSettings);
    }
});

// Job Settings
$('btn-open-folder').addEventListener('click', async () => {
    if (!currentJob) return;
    await api(`/api/jobs/${currentJob}/open-folder`, { method: 'POST' });
});

$('btn-open-image-dir').addEventListener('click', async () => {
    const dir = $('cfg-image-dir').value.trim();
    if (!dir) {
        showToast('Please enter a directory path');
        return;
    }
    const result = await api('/api/system/open-folder', {
        method: 'POST',
        body: { path: dir }
    });
    if (result.error) {
        showToast('Error: ' + result.error);
    }
});

$('btn-clear-logs').addEventListener('click', () => {
    if (!currentJob) return;
    showConfirm('Clear Logs', 'Delete all TensorBoard logs for this job?', async () => {
        await api(`/api/jobs/${currentJob}/clear-logs`, { method: 'POST' });
        showToast('Logs cleared');
    });
});

$('btn-reset-config').addEventListener('click', () => {
    if (!currentJob) return;
    showConfirm('Reset Config', 'Reset all settings to template defaults?', async () => {
        await api(`/api/jobs/${currentJob}/reset-config`, { method: 'POST' });
        selectJob(currentJob);
        showToast('Config reset to defaults');
    });
});

// Close modals on backdrop click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});

// ==========================================
//  Init
// ==========================================

async function init() {
    // 1. FAST LOAD: Apply cached visual settings immediately (Flicker prevention)
    const cachedTheme = localStorage.getItem('ui_theme');
    if (cachedTheme) applyTheme(cachedTheme);

    const cachedBg = localStorage.getItem('ui_background');
    if (cachedBg) {
        try {
            const bg = JSON.parse(cachedBg);
            applyBackground(bg.url, bg.position, bg.dim, bg.brightness, bg.blur, bg.textShadow);
        } catch (e) { }
    }

    // 2. Normal Init
    connectWS();
    await loadJobs();

    // Start status polling
    setInterval(updateGPUActivity, 3000);

    // Watch for config changes
    document.addEventListener('input', (e) => {
        if (e.target.id && e.target.id.startsWith('cfg-')) {
            checkDirty();
        }
    });
    document.addEventListener('change', (e) => {
        if (e.target.id && e.target.id.startsWith('cfg-')) {
            checkDirty();
        }
    });

    // Optimizer custom bindings
    $('cfg-optimizer').addEventListener('change', updateOptimizerOptions);

    // Discard Button
    $('btn-discard').addEventListener('click', discardChanges);

    // Mutual exclusivity for Flash/Sage Attention
    const enforceMutualAttention = (flashId, sageId) => {
        const flash = $(flashId);
        const sage = $(sageId);
        if (!flash || !sage) return;
        flash.addEventListener('change', () => {
            if (flash.checked) sage.checked = false;
            if (flashId.startsWith('gen-')) savePromptTransientSettings();
        });
        sage.addEventListener('change', () => {
            if (sage.checked) flash.checked = false;
            if (flashId.startsWith('gen-')) savePromptTransientSettings();
        });
    };
    enforceMutualAttention('gen-flash-attn', 'gen-sage-attn');


    // Restore Job
    const lastJob = localStorage.getItem('lastJob');
    if (lastJob) {
        const jobExists = Array.from(document.querySelectorAll('.job-item .job-name'))
            .some(el => el.textContent === lastJob);
        if (jobExists) {
            await selectJob(lastJob);
        } else {
            localStorage.removeItem('lastJob');
        }
    }

    // Restore Tab
    const lastTab = localStorage.getItem('lastTab');
    if (lastTab && currentJob) {
        const tabEl = document.querySelector(`.tab[data-tab="${lastTab}"]`);
        if (tabEl) tabEl.click();
    }

    // 3. Sync Settings: Load from server and refresh cache
    const globalConfig = await api('/api/global-config');

    if (globalConfig?.ui?.theme) {
        applyTheme(globalConfig.ui.theme);
    }

    // Apply saved background
    if (globalConfig?.ui?.background) {
        applyBackground(
            globalConfig.ui.background,
            globalConfig.ui.background_position || '50% 50%',
            globalConfig.ui.dim_level ?? 70,
            globalConfig.ui.brightness_level ?? 100,
            globalConfig.ui.blur_level ?? 10,
            globalConfig.ui.text_shadow_size ?? 0
        );
    } else {
        applyBackground('none');
    }
}

init();

window.addEventListener('beforeunload', () => savePromptTransientSettings());
