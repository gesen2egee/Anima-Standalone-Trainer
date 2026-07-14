const express = require('express');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const TOML = require('@iarna/toml');
const net = require('net');
const http = require('http');
const WebSocket = require('ws');
const { createJobQueue } = require('./lib/jobQueue');
const { createQueueCoordinator } = require('./lib/queueCoordinator');
const { findAutoResumeSource } = require('./lib/autoResume');
const { calculateJobProgress } = require('./lib/jobProgress');
const { isSuccessfulTrainingExit } = require('./lib/trainingExit');
const { buildNewJobSubsets } = require('./lib/newJobDataset');
const { buildNewJobSamplePrompts } = require('./lib/newJobSamples');
const { acquireServerInstance } = require('./lib/serverInstance');
const {
    inspectDatasetImageFolders,
    listSdScriptsImages,
    normalizeCaptionExtension,
    resolveDatasetImageFolders
} = require('./lib/datasetImageMatch');

const isWindows = process.platform === 'win32';
const isWSL = process.platform === 'linux' && !!process.env.WSL_DISTRO_NAME;

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Parse CLI argument for port
const args = process.argv.slice(2);
const portArg = args.find(a => a.startsWith('--port='));
const DEFAULT_PORT = portArg
    ? parseInt(portArg.split('=')[1])
    : (parseInt(args[0]) || 3000);
let activePort = DEFAULT_PORT;

// Paths
const ROOT_DIR = path.join(__dirname, '..');
const DEFAULT_JOBS_DIR = path.join(__dirname, 'jobs');
const TEMPLATES_DIR = path.join(__dirname, 'templates');
const GLOBAL_CONFIG_PATH = path.join(__dirname, 'global_config.toml');
const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');
const ARCHITECTURES_PATH = path.join(__dirname, 'architectures.json');
const QUEUE_STATE_PATH = path.join(__dirname, 'queue_state.json');
const TRAINING_PROCESS_STATE_PATH = path.join(__dirname, 'training_process_state.json');
const SERVER_INSTANCE_LOCK_PATH = path.join(__dirname, 'server_instance.json');
const QUEUE_EVENT_LOG_PATH = path.join(__dirname, 'queue_events.log');

let serverInstanceHandle = acquireServerInstance({
    lockPath: SERVER_INSTANCE_LOCK_PATH,
    port: DEFAULT_PORT
});
if (!serverInstanceHandle.acquired) {
    const existing = serverInstanceHandle.existing || {};
    console.error(`\nERROR: Training UI is already running (PID ${existing.pid || 'unknown'}, port ${existing.port || DEFAULT_PORT}).`);
    process.exit(1);
}

require('./lib/setup').runSetup();

let lastQueueEventSignature = '';
let lastQueueEventAt = 0;

function recordQueueEvent(type, details = {}) {
    const now = Date.now();
    const signature = JSON.stringify({ type, ...details });
    if (signature === lastQueueEventSignature && now - lastQueueEventAt < 60000) return;
    lastQueueEventSignature = signature;
    lastQueueEventAt = now;

    const entry = { time: new Date(now).toISOString(), type, ...details };
    try {
        fs.appendFileSync(QUEUE_EVENT_LOG_PATH, `${JSON.stringify(entry)}\n`, 'utf8');
    } catch (err) {
        console.warn(`[Queue] Failed to write event log: ${err.message}`);
    }
    console.log(`[Queue] ${type}`, details);
}

// Load architecture registry
const ARCH_REGISTRY = JSON.parse(fs.readFileSync(ARCHITECTURES_PATH, 'utf8'));

// Resolve architecture from a job config's network_module
function getArchForJob(jobConfig) {
    const explicitArchitecture = jobConfig?.ui_arguments?.architecture || jobConfig?.model_architecture;
    if (explicitArchitecture && ARCH_REGISTRY.architectures[explicitArchitecture]) {
        const explicit = ARCH_REGISTRY.architectures[explicitArchitecture];
        return { id: explicitArchitecture, ...explicit };
    }
    // An explicit Krea2 section takes precedence over shared network names
    // (lora_anima, LoKR, CDKA, and KRONA can all target Krea2 now).
    if (jobConfig?.krea2_arguments || jobConfig?.model_arguments?.krea2_text_encoder) {
        const krea2 = ARCH_REGISTRY.architectures.krea2;
        return { id: 'krea2', ...krea2 };
    }
    const netModule = jobConfig?.network_arguments?.network_module || '';
    for (const [archId, arch] of Object.entries(ARCH_REGISTRY.architectures)) {
        if (arch.network_modules.includes(netModule)) {
            return { id: archId, ...arch };
        }
    }
    // Check for architecture-specific training sections as fallback
    for (const [archId, arch] of Object.entries(ARCH_REGISTRY.architectures)) {
        if (arch.training_section && jobConfig[arch.training_section]) {
            return { id: archId, ...arch };
        }
    }
    // Default
    const defaultId = ARCH_REGISTRY.default_architecture;
    return { id: defaultId, ...ARCH_REGISTRY.architectures[defaultId] };
}

// Build model_arguments from registry + global config for a given architecture
function buildModelArgs(arch, globalConfig) {
    const modelArgs = {};
    for (const [configKey, pathDef] of Object.entries(arch.global_paths)) {
        modelArgs[pathDef.cli_flag] = globalConfig.model_paths?.[configKey] || '';
    }
    return modelArgs;
}

function applyArchitectureJobDefaults(config, architectureId) {
    const architecture = ARCH_REGISTRY.architectures[architectureId];
    if (!architecture?.job_defaults) return;
    for (const [section, defaults] of Object.entries(architecture.job_defaults)) {
        config[section] = { ...(config[section] || {}), ...defaults };
    }
}

function stripUiOnlyBackendArgs(trainingArgs) {
    if (!trainingArgs) return;

    if (trainingArgs.flash_attn) {
        trainingArgs.attn_mode = 'flash';
    }

    [
        'flash_attn',
        'multigpu_mode',
        'use_fsdp',
        'use_cuda_direct',
        'ddp_gradient_as_bucket_view',
        'ddp_static_graph',
        'tp_degree',
        'tp_backend',
        'sequence_parallel',
        'no_fuse_qkv',
        'fsdp_sharding_strategy',
        'fsdp_offload_params',
        'fsdp_reshard_after_forward',
        'fsdp_activation_checkpointing',
        'fsdp_cpu_ram_efficient_loading',
        'fsdp_backward_prefetch',
        'fsdp_forward_prefetch',
        'fsdp_use_orig_params',
        'fsdp_limit_all_gathers',
        'fsdp_auto_wrap_policy',
        'fsdp_min_num_params',
        'fsdp_transformer_layer_cls_to_wrap',
        'fsdp2_reshard_after_forward',
        'fsdp2_offload_params',
        'fsdp2_activation_checkpointing',
        'fsdp2_cpu_ram_efficient_loading',
        'fsdp2_auto_wrap_policy',
        'fsdp2_min_num_params',
        'fsdp2_transformer_layer_cls_to_wrap',
        'deepspeed',
        'zero_stage',
        'offload_optimizer_device',
        'offload_optimizer_nvme_path',
        'offload_param_device',
        'offload_param_nvme_path',
        'zero3_init_flag',
        'zero3_save_16bit_model',
        'fp16_master_weights_and_gradients',
        'step_profile',
        'profile_microbatch'
    ].forEach(key => delete trainingArgs[key]);
}

function normalizeAnimaArgs(merged) {
    const animaArgs = merged.anima_arguments;
    if (!animaArgs) return;

    if (animaArgs.timestep_sample_method && !animaArgs.timestep_sampling) {
        const legacyMethod = animaArgs.timestep_sample_method;
        animaArgs.timestep_sampling = legacyMethod === 'logit_normal' ? 'sigmoid' : legacyMethod;
    }
    delete animaArgs.timestep_sample_method;
}

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Ensure directories exist
if (!fs.existsSync(DEFAULT_JOBS_DIR)) {
    fs.mkdirSync(DEFAULT_JOBS_DIR, { recursive: true });
}
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Ensure global config exists from template
const TEMPLATE_CONFIG_PATH = path.join(__dirname, 'global_config.template.toml');
if (!fs.existsSync(GLOBAL_CONFIG_PATH) && fs.existsSync(TEMPLATE_CONFIG_PATH)) {
    try {
        fs.copyFileSync(TEMPLATE_CONFIG_PATH, GLOBAL_CONFIG_PATH);
        console.log("Created global_config.toml from template.");
    } catch (e) {
        console.error("Failed to create global_config.toml from template:", e);
    }
}

// Track running processes
const runningJobs = new Map();
let detectedTrainingCache = { time: 0, jobs: new Map() };
let detectedTrainingRefreshPromise = null;
const DETECTED_TRAINING_CACHE_MS = 5000;

function invalidateDetectedTrainingProcesses() {
    detectedTrainingCache = { time: 0, jobs: new Map() };
}

// WebSocket clients per job
const wsClients = new Map(); // jobName -> Set<ws>

// --- Helper Functions ---

function sanitizeName(name) {
    let safe = name.replace(/[<>:"/\\|?*]/g, '').trim();
    if (!safe) safe = 'job_' + Date.now();
    return safe;
}

function stripQuotes(p) {
    if (typeof p !== 'string') return p;
    return p.replace(/^['"]+|['"]+$/g, '');
}

function normalizeCaptionPrefixFromTriggerWords(value) {
    const triggerWords = String(value || '').trim();
    if (!triggerWords) return '';
    return /,\s*$/.test(triggerWords) ? triggerWords : `${triggerWords},`;
}

function normalizeCustomCliArgs(value) {
    return String(value || '').replace(/[\r\n]+/g, ' ').trim();
}

function isPidAlive(pid) {
    const numericPid = Number.parseInt(pid, 10);
    if (!Number.isFinite(numericPid) || numericPid <= 0) return false;
    try {
        process.kill(numericPid, 0);
        return true;
    } catch (_) {
        return false;
    }
}

function readTrainingProcessState() {
    try {
        if (!fs.existsSync(TRAINING_PROCESS_STATE_PATH)) return {};
        return JSON.parse(fs.readFileSync(TRAINING_PROCESS_STATE_PATH, 'utf8'));
    } catch (_) {
        return {};
    }
}

function writeTrainingProcessState(jobName, data) {
    const state = readTrainingProcessState();
    state[jobName] = data;
    fs.writeFileSync(TRAINING_PROCESS_STATE_PATH, JSON.stringify(state, null, 2), 'utf8');
}

function clearTrainingProcessState(jobName) {
    const state = readTrainingProcessState();
    if (!state[jobName]) return;
    delete state[jobName];
    fs.writeFileSync(TRAINING_PROCESS_STATE_PATH, JSON.stringify(state, null, 2), 'utf8');
}

function getStoredTrainingProcess(jobName) {
    const stored = readTrainingProcessState()[jobName];
    if (!stored) return null;
    if (!isPidAlive(stored.pid)) {
        clearTrainingProcessState(jobName);
        return null;
    }
    return { ...stored, source: 'state' };
}

function parseMergedConfigPathFromCommandLine(commandLine) {
    const match = String(commandLine || '').match(/--config_file=\\?"?(.+?_merged_config\.toml)\\?"?/i);
    return match ? match[1].replace(/\\"/g, '"').replace(/^"+|"+$/g, '') : '';
}

function parseDetectedTrainingProcesses(raw, windowsMode = isWindows) {
    const jobs = new Map();
    try {
        const processes = windowsMode
            ? (raw ? JSON.parse(raw) : [])
            : raw.split(/\r?\n/).filter(Boolean).map(line => {
                const match = line.trim().match(/^(\d+)\s+(.+)$/);
                return match ? { ProcessId: Number.parseInt(match[1], 10), CommandLine: match[2] } : null;
            }).filter(Boolean);

        (Array.isArray(processes) ? processes : [processes]).forEach(proc => {
            const commandLine = proc?.CommandLine || '';
            const mergedConfigPath = parseMergedConfigPathFromCommandLine(commandLine);
            if (!mergedConfigPath) return;
            const jobName = path.basename(path.dirname(mergedConfigPath));
            if (!jobName) return;
            const pid = Number.parseInt(proc.ProcessId || proc.PID || proc.pid, 10);
            if (!Number.isFinite(pid) || pid <= 0) return;
            jobs.set(jobName, {
                jobName,
                pid,
                commandLine,
                mergedConfigPath,
                source: 'process'
            });
        });
    } catch (_) { }
    return jobs;
}

function refreshDetectedTrainingProcesses() {
    if (detectedTrainingRefreshPromise) return detectedTrainingRefreshPromise;

    detectedTrainingRefreshPromise = new Promise((resolve) => {
        let child;
        let stdout = '';

        try {
            if (isWindows) {
                const script = [
                    '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)',
                    'Get-CimInstance Win32_Process -Filter "CommandLine LIKE \'%_merged_config.toml%\'" |',
                    'Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
                ].join('\n');
                child = spawn('powershell.exe', ['-NoProfile', '-Command', script], {
                    windowsHide: true,
                    stdio: ['ignore', 'pipe', 'ignore']
                });
            } else {
                child = spawn('ps', ['-eo', 'pid=,args='], {
                    stdio: ['ignore', 'pipe', 'ignore']
                });
            }
        } catch (_) {
            detectedTrainingRefreshPromise = null;
            resolve(detectedTrainingCache.jobs);
            return;
        }

        const timer = setTimeout(() => {
            try { child.kill(); } catch (_) { }
        }, 5000);

        child.stdout.on('data', chunk => { stdout += chunk.toString('utf8'); });
        child.on('error', () => {
            clearTimeout(timer);
            detectedTrainingRefreshPromise = null;
            resolve(detectedTrainingCache.jobs);
        });
        child.on('close', () => {
            clearTimeout(timer);
            const jobs = parseDetectedTrainingProcesses(stdout);
            detectedTrainingCache = { time: Date.now(), jobs };
            detectedTrainingRefreshPromise = null;
            resolve(jobs);
        });
    });

    return detectedTrainingRefreshPromise;
}

function getDetectedTrainingProcesses({ refresh = true } = {}) {
    const stale = Date.now() - detectedTrainingCache.time > DETECTED_TRAINING_CACHE_MS;
    if (refresh && stale) refreshDetectedTrainingProcesses().catch(() => { });
    return detectedTrainingCache.jobs;
}

async function getDetectedTrainingProcessesFresh() {
    const stale = Date.now() - detectedTrainingCache.time > DETECTED_TRAINING_CACHE_MS;
    if (!stale) return detectedTrainingCache.jobs;
    return refreshDetectedTrainingProcesses();
}

function getCachedOrStoredTrainingProcess(jobName) {
    const stored = getStoredTrainingProcess(jobName);
    if (stored) return stored;
    return getDetectedTrainingProcesses({ refresh: false }).get(jobName) || null;
}

async function getFreshTrainingProcessInfo(jobName) {
    const memoryJob = runningJobs.get(jobName);
    if (memoryJob?.type === 'training') return { ...memoryJob, jobName, source: 'memory' };
    const stored = getStoredTrainingProcess(jobName);
    if (stored) return stored;
    const detected = await getDetectedTrainingProcessesFresh();
    return detected.get(jobName) || null;
}

function getTrainingProcessInfo(jobName) {
    const memoryJob = runningJobs.get(jobName);
    if (memoryJob?.type === 'training') return { ...memoryJob, jobName, source: 'memory' };
    return getCachedOrStoredTrainingProcess(jobName);
}

async function isJobTrainingFresh(jobName) {
    return !!(await getFreshTrainingProcessInfo(jobName));
}

function isJobTraining(jobName) {
    return !!getTrainingProcessInfo(jobName);
}

async function getRunningTrainingJobNameFresh() {
    const runningTraining = getRunningTrainingJobName();
    if (runningTraining) return runningTraining;
    const detected = await getDetectedTrainingProcessesFresh();
    for (const [name] of detected.entries()) {
        return name;
    }
    return null;
}

async function inspectImageFolder(folderPath, captionExtension = '.txt') {
    return inspectDatasetImageFolders({
        imageDir: folderPath,
        captionExtension,
        batchImport: false,
        toNativePath: p => toNativePath(stripQuotes(String(p || '').trim()))
    });
}

function selectFolderDialog(initialPath = '') {
    return new Promise((resolve, reject) => {
        if (!isWindows && !isWSL) return resolve('');
        const command = String.raw`
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding
$initialPath = [Environment]::GetEnvironmentVariable("ANIMA_INITIAL_FOLDER")
$code = @'
using System;
using System.Runtime.InteropServices;

[ComImport]
[Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
class FileOpenDialog { }

[ComImport]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
[Guid("42f85136-db7e-439c-85f1-e4075d135fc8")]
interface IFileOpenDialog
{
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint pfos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
    void AddPlace(IShellItem psi, int fdap);
    void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
    void Close(int hr);
    void SetClientGuid(ref Guid guid);
    void ClearClientData();
    void SetFilter(IntPtr pFilter);
}

[ComImport]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
[Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe")]
interface IShellItem
{
    void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, out IntPtr ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
}

public class CommonFolderPicker
{
    const uint FOS_PICKFOLDERS = 0x00000020;
    const uint FOS_FORCEFILESYSTEM = 0x00000040;
    const uint FOS_NOCHANGEDIR = 0x00000008;
    const uint FOS_PATHMUSTEXIST = 0x00000800;
    const uint SIGDN_FILESYSPATH = 0x80058000;

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string pszPath,
        IntPtr pbc,
        ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

    static void ApplyInitialFolder(IFileOpenDialog dialog, string initialPath)
    {
        if (String.IsNullOrWhiteSpace(initialPath) || !System.IO.Directory.Exists(initialPath)) return;

        try
        {
            Guid shellItemGuid = new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");
            IShellItem folder;
            SHCreateItemFromParsingName(initialPath, IntPtr.Zero, ref shellItemGuid, out folder);
            dialog.SetDefaultFolder(folder);
            dialog.SetFolder(folder);
        }
        catch
        {
            // Ignore invalid or unavailable initial folders and let the dialog choose its default.
        }
    }

    public static string Pick(string initialPath)
    {
        IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialog();
        uint options;
        dialog.GetOptions(out options);
        dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR | FOS_PATHMUSTEXIST);
        dialog.SetTitle("Select image folder");
        ApplyInitialFolder(dialog, initialPath);
        int hr = dialog.Show(IntPtr.Zero);
        if (hr != 0) return "";

        IShellItem item;
        dialog.GetResult(out item);
        IntPtr pathPtr;
        item.GetDisplayName(SIGDN_FILESYSPATH, out pathPtr);
        string path = Marshal.PtrToStringUni(pathPtr);
        Marshal.FreeCoTaskMem(pathPtr);
        return path;
    }
}
'@
Add-Type -TypeDefinition $code
$selected = [CommonFolderPicker]::Pick($initialPath)
if (-not [string]::IsNullOrWhiteSpace($selected)) {
    [Console]::WriteLine($selected)
}
`;
        const exe = 'powershell.exe';
        const child = spawn(exe, ['-NoProfile', '-STA', '-Command', command], {
            windowsHide: false,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: {
                ...process.env,
                ANIMA_INITIAL_FOLDER: String(initialPath || '').trim()
            }
        });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', chunk => { stdout += chunk.toString('utf8'); });
        child.stderr.on('data', chunk => { stderr += chunk.toString('utf8'); });
        child.on('error', reject);
        child.on('close', code => {
            const selectedPath = stdout.trim();
            if (selectedPath) return resolve(selectedPath);
            if (code === 0) return resolve('');
            reject(new Error(stderr.trim() || `Folder picker exited with code ${code}`));
        });
    });
}

function getJobPath(name) {
    return path.join(getJobsDir(), sanitizeName(name));
}

function getGlobalConfig() {
    if (fs.existsSync(GLOBAL_CONFIG_PATH)) {
        try {
            const config = TOML.parse(fs.readFileSync(GLOBAL_CONFIG_PATH, 'utf8'));
            return config;
        } catch (err) {
            console.error('Failed to parse global config:', err.message);
        }
    }
    return {
        model_paths: {
            // Anima
            dit_path: '',
            qwen3_path: '',
            vae_path: '',
            // Lumina
            lumina_dit_path: '',
            gemma2_path: '',
            lumina_vae_path: '',
            // Krea 2
            krea2_dit_path: '',
            krea2_text_encoder_path: '',
            krea2_vae_path: ''
        },
        venv_path: path.join(ROOT_DIR, 'venv'),
        jobs_dir: DEFAULT_JOBS_DIR
    };
}

function getJobsDir() {
    const globalConfig = getGlobalConfig();
    const configured = stripQuotes(globalConfig.jobs_dir || '');
    const jobsDir = configured ? toNativePath(configured) : DEFAULT_JOBS_DIR;
    if (!fs.existsSync(jobsDir)) {
        fs.mkdirSync(jobsDir, { recursive: true });
    }
    return jobsDir;
}

function jobExists(jobName) {
    const configPath = path.join(getJobPath(jobName), 'config.toml');
    return fs.existsSync(configPath);
}

const trainingQueue = createJobQueue({
    statePath: QUEUE_STATE_PATH,
    jobExists
});
let queueAutoRunning = trainingQueue.getState().autoRunning;

function setQueueAutoRunning(enabled, error = null) {
    queueAutoRunning = enabled === true;
    trainingQueue.setAutoRunning(queueAutoRunning, { error });
}

const queueCoordinator = createQueueCoordinator({
    isEnabled: () => queueAutoRunning,
    getRunningJob: () => getRunningTrainingJobNameFresh(),
    getNextJob: () => trainingQueue.getNext(),
    startJob: (jobName) => startQueuedJob(jobName),
    onQueueEmpty: () => {
        setQueueAutoRunning(false);
    },
    onError: (err) => {
        setQueueAutoRunning(false, err.message);
        console.error(`[Queue] ${err.message}`);
    },
    onTransition: event => recordQueueEvent(`coordinator-${event.type}`, event),
    retryDelayMs: 2000
});

async function reconcileQueue(reason = 'watchdog') {
    if (!queueAutoRunning) return;
    const state = trainingQueue.getState();
    if (state.items.length === 0) {
        setQueueAutoRunning(false);
        return;
    }

    const runningJob = await getRunningTrainingJobNameFresh();
    if (runningJob) return;
    if (state.active) {
        trainingQueue.clearActive(state.active);
        recordQueueEvent('recovered-stale-active', { reason, jobName: state.active });
    }
    queueCoordinator.requestAdvance(0);
}

// Periodically reconcile persisted queue intent. This recovers missed close
// notifications and server restarts without relying on an open browser.
let queueReconcileTimer = null;

function startQueueReconciler() {
    if (queueReconcileTimer) return;
    queueReconcileTimer = setInterval(() => {
        reconcileQueue().catch(err => recordQueueEvent('reconcile-error', { error: err.message }));
    }, 5000);
    queueReconcileTimer.unref?.();
}

function getRunningTrainingJobName() {
    for (const [name, job] of runningJobs.entries()) {
        if (job.type === 'training') return name;
    }
    for (const [name] of getDetectedTrainingProcesses({ refresh: false }).entries()) {
        return name;
    }
    for (const [name, info] of Object.entries(readTrainingProcessState())) {
        if (isPidAlive(info?.pid)) return name;
        clearTrainingProcessState(name);
    }
    return null;
}

function stopTrainingJob(jobName) {
    const job = getTrainingProcessInfo(jobName);
    if (!job) return null;

    const memoryJob = runningJobs.get(jobName);
    if (memoryJob) memoryJob.stopRequested = true;
    job.stopRequested = true;
    broadcastStatus(jobName, 'stopping');

    if (job.pid) {
        killProcess(job.pid, 8000)
            .then(() => clearTrainingProcessState(jobName))
            .catch(() => {});
    }

    return job;
}

async function stopRunningTrainingForQueue() {
    const runningTraining = await getRunningTrainingJobNameFresh();
    if (!runningTraining) {
        const state = trainingQueue.getState();
        if (state.active) trainingQueue.clearActive(state.active);
        return null;
    }

    const stoppedJob = stopTrainingJob(runningTraining);
    trainingQueue.clearActive(runningTraining);
    return stoppedJob ? runningTraining : null;
}

// Serve architecture registry to frontend
app.get('/api/architectures', (req, res) => {
    res.json(ARCH_REGISTRY);
});

app.get('/api/gpu/activity', (req, res) => {
    const activity = {};

    // Check running jobs (training/generation)
    for (const [name, job] of runningJobs.entries()) {
        if (job.gpuIds) {
            job.gpuIds.split(',').forEach(id => {
                const trimmed = id.trim();
                if (trimmed) {
                    activity[trimmed] = job.type === 'generation' ? 'sampling' : 'training';
                }
            });
        }
    }

    // Check persistent generation
    if (persistentGenProcess && persistentGenProcess.gpuIds) {
        persistentGenProcess.gpuIds.split(',').forEach(id => {
            const trimmed = id.trim();
            if (trimmed) activity[trimmed] = 'sampling';
        });
    }

    res.json(activity);
});

// Get GPU Information using nvidia-smi with Python fallback
async function getDetectedGPUs() {
    return new Promise((resolve) => {
        // 1. Try nvidia-smi
        const smi = spawn('nvidia-smi', ['--query-gpu=index,name,memory.total', '--format=csv,noheader']);
        let stdout = '';
        let stderr = '';

        smi.stdout.on('data', (data) => stdout += data);
        smi.stderr.on('data', (data) => stderr += data);

        smi.on('close', (code) => {
            if (code === 0 && stdout) {
                const gpus = stdout.trim().split('\n').map(line => {
                    const parts = line.split(',').map(s => s.trim());
                    if (parts.length < 3) return null;
                    return {
                        index: parseInt(parts[0]),
                        name: parts[1],
                        memory: parts[2]
                    };
                }).filter(g => g !== null);
                return resolve(gpus);
            }

            // 2. Fallback to Python (torch)
            console.warn("nvidia-smi failed, trying python fallback...");
            const globalConfig = getGlobalConfig();
            const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
            let pythonPath = 'python'; // Default
            if (process.platform === 'win32') {
                pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
            } else {
                pythonPath = path.join(venvPath, 'bin', 'python');
            }

            if (!fs.existsSync(pythonPath)) {
                pythonPath = 'python';
            }

            const pyScript = "import torch; import json; print(json.dumps([{'index': i, 'name': torch.cuda.get_device_name(i), 'memory': f'{torch.cuda.get_device_properties(i).total_memory // 1024**2} MiB'} for i in range(torch.cuda.device_count())]))";

            const pyProc = spawn(pythonPath, ['-c', pyScript]);
            let pyOut = '';
            let pyErr = '';

            pyProc.stdout.on('data', (data) => pyOut += data);
            pyProc.stderr.on('data', (data) => pyErr += data);

            pyProc.on('close', (pyCode) => {
                if (pyCode !== 0) {
                    console.error("Python GPU detection failed:", pyErr);
                    return resolve([]);
                }
                try {
                    const gpus = JSON.parse(pyOut.trim());
                    resolve(gpus);
                } catch (e) {
                    console.error("Failed to parse Python GPU output:", e);
                    resolve([]);
                }
            });
        });

        smi.on('error', (err) => {
            // Silently fail to fallback
        });
    });
}

function getDefaultConfig() {
    const templatePath = path.join(TEMPLATES_DIR, 'config_template.toml');
    if (fs.existsSync(templatePath)) {
        try {
            return { config: TOML.parse(fs.readFileSync(templatePath, 'utf8')), useFallback: false };
        } catch (err) {
            console.error('Config template parse error:', err.message);
        }
    }
    return {
        config: {
            training_arguments: {
                output_name: 'my_anima_lora',
                learning_rate: 5e-4,
                max_train_epochs: 20,
                mixed_precision: 'bf16'
            },
            network_arguments: {
                network_module: 'networks.krona',
                network_dim: 16,
                network_alpha: 16
            }
        },
        useFallback: true
    };
}

function getNetworkModuleLearningRate(networkModule) {
    return networkModule === 'networks.cdka' ? 1e-4 : 5e-4;
}

function getDefaultDataset() {
    const templatePath = path.join(TEMPLATES_DIR, 'dataset_template.toml');
    if (fs.existsSync(templatePath)) {
        try {
            return TOML.parse(fs.readFileSync(templatePath, 'utf8'));
        } catch (err) {
            console.error('Dataset template parse error:', err.message);
        }
    }
    return {
        general: { enable_bucket: true },
        datasets: [{ resolution: [1536, 1536], batch_size: 4, caption_extension: '.txt', subsets: [{ image_dir: '', num_repeats: 1 }] }]
    };
}

// Startup validation
(function validateTemplates() {
    const configTemplate = path.join(TEMPLATES_DIR, 'config_template.toml');
    const datasetTemplate = path.join(TEMPLATES_DIR, 'dataset_template.toml');
    [configTemplate, datasetTemplate].forEach(f => {
        if (fs.existsSync(f)) {
            try {
                TOML.parse(fs.readFileSync(f, 'utf8'));
                console.log(`Template validated: ${path.basename(f)}`);
            } catch (err) {
                console.error(`Template error in ${path.basename(f)}: ${err.message}`);
            }
        } else {
            console.warn(`Template not found: ${path.basename(f)}`);
        }
    });
})();

function broadcastLog(jobName, message) {
    const clients = wsClients.get(jobName);
    if (clients) {
        const data = JSON.stringify({ job: jobName, type: 'log', data: message });
        clients.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(data);
            }
        });
    }
}

function broadcastStatus(jobName, status) {
    const clients = wsClients.get(jobName);
    if (clients) {
        const data = JSON.stringify({ job: jobName, type: 'status', data: status });
        clients.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(data);
            }
        });
    }
    broadcastQueueChanged();
}

function broadcastQueueChanged() {
    const data = JSON.stringify({ type: 'queue' });
    wss.clients.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(data);
        }
    });
}

// Build the full TOML config file for training, merging global model paths + job paths
function buildTrainingConfig(jobName, jobPath) {
    const globalConfig = getGlobalConfig();
    const configPath = path.join(jobPath, 'config.toml');
    const jobConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));

    const outputDir = path.join(jobPath, 'output');
    const loggingDir = outputDir;
    const datasetConfigPath = path.join(jobPath, 'dataset.toml');
    const samplePromptsPath = path.join(jobPath, 'sample_prompts.txt');

    // Build the merged config
    const merged = {};

    // Resolve architecture from job config
    const arch = getArchForJob(jobConfig);

    // Model arguments from global config, mapped through registry
    merged.model_arguments = buildModelArgs(arch, globalConfig);

    // Dataset arguments
    merged.dataset_arguments = {
        dataset_config: datasetConfigPath,
        cache_latents_to_disk: jobConfig.training_arguments?.cache_latents_to_disk ?? true,
        cache_text_encoder_outputs_to_disk: jobConfig.training_arguments?.cache_text_encoder_outputs_to_disk ?? true
    };

    // Training arguments (remove cache args since they're in dataset_arguments)
    const trainingArgs = { ...jobConfig.training_arguments };
    delete trainingArgs.cache_latents_to_disk;
    delete trainingArgs.cache_text_encoder_outputs_to_disk;
    merged.training_arguments = {
        ...trainingArgs,
        output_dir: outputDir,
        logging_dir: loggingDir
    };
    if (arch.id === 'krea2') {
        delete merged.training_arguments.cpu_offload_checkpointing;
        delete merged.training_arguments.unsloth_offload_checkpointing;
        merged.training_arguments.torch_compile = false;
    }
    stripUiOnlyBackendArgs(merged.training_arguments);

    const outputName = merged.training_arguments.output_name || jobName;
    const autoResumeUserSet = jobConfig.network_arguments?.auto_resume_last_state_user_set === true;
    const autoResumeEnabled = !(autoResumeUserSet && jobConfig.network_arguments?.auto_resume_last_state === false);
    let autoResumeNetworkWeights = null;

    if (autoResumeEnabled) {
        merged.training_arguments.save_state = true;
    }

    // Move resume from network_args to training_args
    if (jobConfig.network_arguments?.resume) {
        merged.training_arguments.resume = jobConfig.network_arguments.resume;
    }

    // Auto-resume: prefer the latest state, then fall back to the latest LoRA checkpoint.
    if (autoResumeEnabled && !merged.training_arguments.resume) {
        const source = findAutoResumeSource(outputDir, outputName, {
            allowCheckpoint: Boolean(jobConfig.network_arguments?.network_module && !jobConfig.network_arguments?.network_weights)
        });
        if (source?.type === 'state') {
            merged.training_arguments.resume = source.path;
            console.log(`[auto-resume] Detected last state: ${source.path}`);
        } else if (source?.type === 'checkpoint') {
            autoResumeNetworkWeights = source.path;
            console.log(`[auto-resume] Detected last checkpoint: ${source.path}`);
        } else {
            console.log(`[auto-resume] No saved state or checkpoint found in ${outputDir}, starting fresh.`);
        }
    }

    // Add sample prompts if file exists and has content
    if (arch.id !== 'krea2' && fs.existsSync(samplePromptsPath)) {
        const prompts = fs.readFileSync(samplePromptsPath, 'utf8').trim();
        if (prompts.length > 0) {
            const ta = jobConfig.training_arguments || {};
            if (ta.sample_every_n_steps || ta.sample_every_n_epochs || ta.sample_at_first) {
                merged.sample_arguments = {
                    sample_prompts: samplePromptsPath
                };

                // Prefer steps if set. Do not invent an epoch interval for sample_at_first-only jobs.
                if (ta.sample_every_n_steps) {
                    merged.sample_arguments.sample_every_n_steps = ta.sample_every_n_steps;
                } else if (ta.sample_every_n_epochs) {
                    merged.sample_arguments.sample_every_n_epochs = ta.sample_every_n_epochs;
                }
                if (ta.sample_at_first) {
                    merged.sample_arguments.sample_at_first = true;
                }
            }
        }
    }

    delete merged.training_arguments.sample_every_n_epochs;
    delete merged.training_arguments.sample_every_n_steps;
    delete merged.training_arguments.sample_at_first;

    // Convert freeze_llm_adapter flag → llm_adapter_lr: 0 (works for both DDP and TP+SP FFT)
    if (merged.training_arguments.freeze_llm_adapter) {
        merged.training_arguments.llm_adapter_lr = 0;
    }
    delete merged.training_arguments.freeze_llm_adapter;

    // Network arguments
    merged.network_arguments = { ...jobConfig.network_arguments };
    if (autoResumeNetworkWeights && !merged.network_arguments.network_weights) {
        merged.network_arguments.network_weights = autoResumeNetworkWeights;
    }
    delete merged.network_arguments.resume;
    delete merged.network_arguments.auto_resume_last_state;
    delete merged.network_arguments.auto_resume_last_state_user_set;

    // Anima arguments
    if (jobConfig.anima_arguments) {
        merged.anima_arguments = { ...jobConfig.anima_arguments };
        normalizeAnimaArgs(merged);
    }

    // Lumina arguments
    if (jobConfig.lumina_arguments) {
        merged.lumina_arguments = { ...jobConfig.lumina_arguments };
    }

    // Krea 2 uses the same shared DiT argument parser but has its own cache/token settings.
    if (jobConfig.krea2_arguments) {
        merged.krea2_arguments = { ...jobConfig.krea2_arguments };
    }

    return merged;
}

// --- WebSocket ---

// Heartbeat: terminate dead connections so wss.clients never accumulates stale entries
setInterval(() => {
    wss.clients.forEach(ws => {
        if (ws.isAlive === false) { ws.terminate(); return; }
        ws.isAlive = false;
        ws.ping();
    });
}, 30000);

wss.on('connection', (ws) => {
    ws.isAlive = true;
    ws.on('pong', () => { ws.isAlive = true; });
    ws.subscribedJob = null;

    ws.on('message', (message) => {
        try {
            const msg = JSON.parse(message);
            if (msg.type === 'subscribe' && msg.job) {
                // Unsubscribe from previous
                if (ws.subscribedJob) {
                    const oldClients = wsClients.get(ws.subscribedJob);
                    if (oldClients) oldClients.delete(ws);
                }
                // Subscribe to new
                ws.subscribedJob = msg.job;
                if (!wsClients.has(msg.job)) {
                    wsClients.set(msg.job, new Set());
                }
                wsClients.get(msg.job).add(ws);

                // Send current status
                const isRunning = runningJobs.has(msg.job);
                ws.send(JSON.stringify({
                    job: msg.job,
                    type: 'status',
                    data: isRunning ? 'running' : 'idle'
                }));

                // Send buffered logs
                const jobData = runningJobs.get(msg.job);
                if (jobData && jobData.logBuffer) {
                    ws.send(JSON.stringify({
                        job: msg.job,
                        type: 'log',
                        data: jobData.logBuffer.join('')
                    }));
                }
            }
        } catch (e) {
            // ignore
        }
    });

    ws.on('close', () => {
        if (ws.subscribedJob) {
            const clients = wsClients.get(ws.subscribedJob);
            if (clients) clients.delete(ws);
        }
    });
});

// --- Global Config API ---

app.get('/api/global-config', (req, res) => {
    try {
        res.json(getGlobalConfig());
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.put('/api/global-config', (req, res) => {
    try {
        const body = req.body;
        if (body.model_paths) {
            for (const key in body.model_paths) {
                body.model_paths[key] = stripQuotes(body.model_paths[key]);
            }
        }
        const tomlStr = TOML.stringify(body);
        fs.writeFileSync(GLOBAL_CONFIG_PATH, tomlStr, 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Background Image API ---

app.post('/api/global/background', (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ error: 'No image data' });

        const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
        const extension = image.split(';')[0].split('/')[1];
        const filename = `bg_${Date.now()}.${extension}`;
        const filePath = path.join(UPLOAD_DIR, filename);

        // Delete old backgrounds
        if (fs.existsSync(UPLOAD_DIR)) {
            fs.readdirSync(UPLOAD_DIR).forEach(file => fs.unlinkSync(path.join(UPLOAD_DIR, file)));
        }

        fs.writeFileSync(filePath, base64Data, 'base64');
        res.json({ success: true, url: `/uploads/${filename}` });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.delete('/api/global/background', (req, res) => {
    try {
        if (fs.existsSync(UPLOAD_DIR)) {
            fs.readdirSync(UPLOAD_DIR).forEach(file => fs.unlinkSync(path.join(UPLOAD_DIR, file)));
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- System API Routes ---

app.get('/api/system/gpus', async (req, res) => {
    const gpus = await getDetectedGPUs();
    res.json(gpus);
});

// --- Job API Routes ---

// List all jobs
app.get('/api/jobs', async (req, res) => {
    try {
        await getDetectedTrainingProcessesFresh();
        const jobsDir = getJobsDir();
        if (!fs.existsSync(jobsDir)) return res.json([]);
        const queueState = trainingQueue.getState();
        const queuedJobs = new Set(queueState.items);
        const jobs = fs.readdirSync(jobsDir, { withFileTypes: true })
            .filter(d => d.isDirectory())
            .map(d => {
                const configPath = path.join(jobsDir, d.name, 'config.toml');
                const hasConfig = fs.existsSync(configPath);
                let mtime = 0;
                let config = {};
                if (hasConfig) {
                    try {
                        mtime = fs.statSync(configPath).mtimeMs;
                        config = TOML.parse(fs.readFileSync(configPath, 'utf8'));
                    } catch (e) { }
                }
                const trainingArgs = config.training_arguments || {};
                return {
                    name: d.name,
                    hasConfig,
                    running: isJobTraining(d.name),
                    queued: queuedJobs.has(d.name),
                    queueIndex: queueState.items.indexOf(d.name),
                    queueActive: queueState.active === d.name,
                    progress: calculateJobProgress(path.join(jobsDir, d.name, 'output'), {
                        outputName: trainingArgs.output_name || d.name,
                        maxTrainSteps: trainingArgs.max_train_steps,
                        maxTrainEpochs: trainingArgs.max_train_epochs
                    }),
                    mtime
                };
            })
            .sort((a, b) => b.mtime - a.mtime);
        res.json(jobs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Create new job
app.post('/api/jobs', (req, res) => {
    try {
        const {
            name,
            output_name,
            network_module,
            image_dir,
            max_train_steps,
            model_architecture,
            trigger_words,
            generate_samples,
            batch_import,
            auto_balance_repeats
        } = req.body;
        if (!name) return res.status(400).json({ error: 'Name required' });

        const safeName = sanitizeName(name);
        const jobPath = path.join(getJobsDir(), safeName);

        if (fs.existsSync(jobPath)) {
            return res.status(409).json({ error: 'Job already exists' });
        }

        // Create directory structure
        fs.mkdirSync(jobPath, { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'output'), { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'logs'), { recursive: true });
        fs.mkdirSync(path.join(jobPath, 'samples'), { recursive: true });

        // Copy template configs
        const { config, useFallback } = getDefaultConfig();
        config.ui_arguments = config.ui_arguments || {};
        if (model_architecture && ARCH_REGISTRY.architectures[model_architecture]) {
            config.ui_arguments.architecture = model_architecture;
            applyArchitectureJobDefaults(config, model_architecture);
        }
        config.training_arguments = config.training_arguments || {};
        if (output_name && String(output_name).trim()) {
            config.training_arguments.output_name = sanitizeName(String(output_name));
        }
        const parsedSteps = Number.parseInt(max_train_steps, 10);
        if (Number.isFinite(parsedSteps) && parsedSteps > 0) {
            config.training_arguments.max_train_steps = parsedSteps;
            delete config.training_arguments.max_train_epochs;
        }
        const supportedNetworkModules = [
            'networks.lora_anima', 'networks.lora_lumina', 'networks.lora_krea2',
            'networks.lora', 'networks.lokr', 'networks.cdka', 'networks.krona'
        ];
        if (network_module && supportedNetworkModules.includes(network_module)) {
            config.network_arguments = config.network_arguments || {};
            config.network_arguments.network_module = network_module;
        }
        if (generate_samples !== true) {
            delete config.training_arguments.sample_at_first;
            delete config.training_arguments.sample_every_n_steps;
        }
        fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(config), 'utf8');

        const datasetConfig = getDefaultDataset();
        const triggerCaptionPrefix = normalizeCaptionPrefixFromTriggerWords(trigger_words);
        const shouldConfigureDataset = (image_dir && String(image_dir).trim()) || triggerCaptionPrefix;
        let datasetMatch = null;
        if (shouldConfigureDataset) {
            const datasets = Array.isArray(datasetConfig.datasets)
                ? datasetConfig.datasets
                : datasetConfig.datasets
                    ? [datasetConfig.datasets]
                    : [{}];
            const firstDataset = datasets[0] || {};
            const subsets = Array.isArray(firstDataset.subsets)
                ? firstDataset.subsets
                : firstDataset.subsets
                    ? [firstDataset.subsets]
                    : [{}];
            datasetMatch = resolveDatasetImageFolders({
                imageDir: image_dir,
                batchImport: batch_import === true,
                autoBalanceRepeats: auto_balance_repeats === true,
                toNativePath: p => toNativePath(stripQuotes(String(p || '').trim()))
            });
            const generatedSubsets = buildNewJobSubsets({
                imageDir: image_dir,
                triggerCaptionPrefix,
                batchImport: batch_import === true,
                autoBalanceRepeats: auto_balance_repeats === true,
                baseSubset: subsets[0] || {},
                toNativePath: p => toNativePath(stripQuotes(String(p || '').trim()))
            });
            firstDataset.subsets = generatedSubsets.length > 0 ? generatedSubsets : subsets;
            datasets[0] = firstDataset;
            datasetConfig.datasets = datasets;
        }
        fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(datasetConfig), 'utf8');

        // Generate two sample prompts when requested; otherwise keep the template behavior.
        const promptsTemplate = path.join(TEMPLATES_DIR, 'sample_prompts.txt');
        const generatedSamplePrompts = generate_samples === true
            ? buildNewJobSamplePrompts(trigger_words)
            : '';
        if (generatedSamplePrompts) {
            fs.writeFileSync(path.join(jobPath, 'sample_prompts.txt'), `${generatedSamplePrompts}\n`, 'utf8');
        } else if (fs.existsSync(promptsTemplate)) {
            fs.copyFileSync(promptsTemplate, path.join(jobPath, 'sample_prompts.txt'));
        } else {
            fs.writeFileSync(path.join(jobPath, 'sample_prompts.txt'), '', 'utf8');
        }

        res.json({
            name: safeName,
            path: jobPath,
            useFallback,
            dataset_match: datasetMatch ? {
                mode: datasetMatch.mode,
                matched_folder_count: datasetMatch.matchedFolderCount,
                image_count: datasetMatch.totalImages,
                folders: datasetMatch.folders.map(folder => ({
                    image_dir: folder.imageDir,
                    image_count: folder.imageCount,
                    repeats: folder.repeats,
                    trigger_words: folder.triggerWords
                }))
            } : null
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Get job config
app.get('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const configPath = path.join(jobPath, 'config.toml');
        const datasetPath = path.join(jobPath, 'dataset.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const config = TOML.parse(fs.readFileSync(configPath, 'utf8'));
        const dataset = fs.existsSync(datasetPath)
            ? TOML.parse(fs.readFileSync(datasetPath, 'utf8'))
            : getDefaultDataset();

        res.json({ name: req.params.name, config, dataset });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/jobs/:name/cli-command', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const jobPath = getJobPath(jobName);
        if (!fs.existsSync(jobPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const mergedConfig = buildTrainingConfig(jobName, jobPath);
        if (isWSL) {
            Object.assign(mergedConfig, convertPathsInObject(mergedConfig));
        }
        const mergedConfigPath = path.join(jobPath, '_merged_config.toml');
        const launch = buildTrainingLaunchCommand(jobName, jobPath, mergedConfig, mergedConfigPath);
        if (launch.error) return res.status(400).json({ error: launch.error });
        const datasetPath = path.join(jobPath, 'dataset.toml');
        const samplePromptsPath = path.join(jobPath, 'sample_prompts.txt');
        const datasetConfig = fs.existsSync(datasetPath)
            ? TOML.parse(fs.readFileSync(datasetPath, 'utf8'))
            : getDefaultDataset();
        const previewDataset = isWSL ? convertPathsInObject(datasetConfig) : datasetConfig;

        res.json({
            command: launch.trainScript,
            base_command: launch.baseTrainScript,
            toml: TOML.stringify(mergedConfig),
            dataset_toml: TOML.stringify(previewDataset),
            sample_prompts: fs.existsSync(samplePromptsPath) ? fs.readFileSync(samplePromptsPath, 'utf8') : '',
            custom_cli_args: launch.customCliArgs
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Update job config
app.put('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (!fs.existsSync(jobPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        if (req.body.config) {
            const config = req.body.config;
            const na = config.network_arguments;
            if (na) {
                if (na.resume)          na.resume          = stripQuotes(na.resume);
                if (na.network_weights) na.network_weights = stripQuotes(na.network_weights);
            }
            fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(config), 'utf8');
        }
        if (req.body.dataset) {
            fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(req.body.dataset), 'utf8');
        }

        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

function toArrayConfig(value) {
    if (!value) return [];
    return Array.isArray(value) ? value : [value];
}

function summarizeMatchedFolders(folders) {
    return folders.map(folder => ({
        image_dir: folder.imageDir,
        image_count: folder.imagePaths.length
    }));
}

function resolveTaggerTargetsForJob(jobPath, requestedImageDir = '') {
    const requested = stripQuotes(String(requestedImageDir || '').trim());
    const folders = [];

    if (requested) {
        const nativeDir = toNativePath(requested);
        folders.push({
            imageDir: requested,
            nativeImageDir: nativeDir,
            imagePaths: listSdScriptsImages(nativeDir)
        });
    } else {
        const datasetPath = path.join(jobPath, 'dataset.toml');
        if (!fs.existsSync(datasetPath)) {
            return { imagePaths: [], folders: [], matchedFolderCount: 0 };
        }
        const datasetConfig = TOML.parse(fs.readFileSync(datasetPath, 'utf8'));
        for (const dataset of toArrayConfig(datasetConfig.datasets)) {
            for (const subset of toArrayConfig(dataset.subsets)) {
                if (!subset?.image_dir) continue;
                const imageDir = stripQuotes(String(subset.image_dir).trim());
                const nativeDir = toNativePath(imageDir);
                const imagePaths = listSdScriptsImages(nativeDir);
                if (imagePaths.length < 1) continue;
                folders.push({ imageDir, nativeImageDir: nativeDir, imagePaths });
            }
        }
    }

    const seen = new Set();
    const imagePaths = [];
    for (const folder of folders) {
        for (const imagePath of folder.imagePaths) {
            if (seen.has(imagePath)) continue;
            seen.add(imagePath);
            imagePaths.push(imagePath);
        }
    }

    return {
        imagePaths,
        folders: folders.filter(folder => folder.imagePaths.length > 0),
        matchedFolderCount: folders.filter(folder => folder.imagePaths.length > 0).length
    };
}

app.post('/api/jobs/:name/datasets/split-timesteps', async (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (!fs.existsSync(jobPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }
        const { image_dir, trigger_word } = req.body;
        if (!image_dir) {
            return res.status(400).json({ error: 'Image directory required' });
        }

        const cleanImgDir = stripQuotes(String(image_dir || '').trim());
        const nativeImgDir = toNativePath(cleanImgDir);
        if (!fs.existsSync(nativeImgDir) || !fs.statSync(nativeImgDir).isDirectory()) {
            return res.status(400).json({ error: `Image directory not found: ${nativeImgDir}` });
        }

        // 初始化進度檔案
        const progressPath = path.join(jobPath, 'split_progress.json');
        fs.writeFileSync(progressPath, JSON.stringify({ current: 0, total: 100, status: 'Initializing classify model...' }), 'utf8');

        // 異步啟動 Python 分類進程
        const { spawn } = require('child_process');
        const globalConfig = getGlobalConfig();
        const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
        const venv = getVenvPaths(venvPath);
        const scriptPath = path.join(ROOT_DIR, 'library/classify_timesteps.py');

        // 檢查 image_dir 底下是否有「數字_名稱」子資料夾
        const hasNumberedSubdirs = fs.readdirSync(nativeImgDir, { withFileTypes: true })
            .some(entry => entry.isDirectory() && /^\d+_(.+)$/.test(entry.name));

        const args = [
            scriptPath,
            '--image_dir', nativeImgDir,
            '--trigger_word', trigger_word || 'miku',
            '--job_dir', jobPath
        ];
        if (hasNumberedSubdirs) {
            args.push('--batch_import');
        }

        console.log(`Running auto split: ${venv.python} ${args.join(' ')}`);
        const child = spawn(venv.python, args, {
            cwd: ROOT_DIR,
            env: {
                ...process.env,
                PYTHONIOENCODING: 'utf-8',
                PYTHONUTF8: '1',
            },
            windowsHide: true,
        });

        child.stdout.on('data', (data) => {
            console.log(`[split-stdout]: ${data}`);
        });

        child.stderr.on('data', (data) => {
            console.error(`[split-stderr]: ${data}`);
        });

        child.on('close', (code) => {
            console.log(`classify_timesteps.py exited with code ${code}`);
            if (code === 0) {
                try {
                    const datasetPath = path.join(jobPath, 'dataset.toml');
                    if (fs.existsSync(datasetPath)) {
                        const datasetConfig = TOML.parse(fs.readFileSync(datasetPath, 'utf8'));
                        const datasets = Array.isArray(datasetConfig.datasets) ? datasetConfig.datasets : [datasetConfig.datasets || {}];
                        const firstDataset = datasets[0] || {};
                        const subsets = Array.isArray(firstDataset.subsets) ? firstDataset.subsets : [firstDataset.subsets || {}];
                        
                        const resolvedSubsets = buildNewJobSubsets({
                            imageDir: cleanImgDir,
                            triggerCaptionPrefix: trigger_word ? normalizeCaptionPrefixFromTriggerWords(trigger_word) : '',
                            batchImport: true,
                            autoBalanceRepeats: false,
                            baseSubset: subsets[0] || {},
                            toNativePath: p => toNativePath(stripQuotes(String(p || '').trim()))
                        });

                        const cleanTarget = stripQuotes(String(image_dir || '').trim()).toLowerCase();
                        const newSubsets = [];
                        let replaced = false;
                        for (const sub of subsets) {
                            const cleanSubDir = stripQuotes(String(sub.image_dir || '').trim()).toLowerCase();
                            if (cleanSubDir === cleanTarget) {
                                newSubsets.push(...resolvedSubsets);
                                replaced = true;
                            } else {
                                newSubsets.push(sub);
                            }
                        }
                        
                        if (!replaced) {
                            const validSubsets = subsets.filter(sub => {
                                if (!sub.image_dir) return false;
                                return fs.existsSync(toNativePath(stripQuotes(String(sub.image_dir).trim())));
                            });
                            const existingDirs = new Set(validSubsets.map(s => stripQuotes(String(s.image_dir).trim()).toLowerCase()));
                            for (const newSub of resolvedSubsets) {
                                const newDirClean = stripQuotes(String(newSub.image_dir).trim()).toLowerCase();
                                if (!existingDirs.has(newDirClean)) {
                                    validSubsets.push(newSub);
                                }
                            }
                            firstDataset.subsets = validSubsets;
                        } else {
                            firstDataset.subsets = newSubsets;
                        }

                        datasets[0] = firstDataset;
                        datasetConfig.datasets = datasets;
                        fs.writeFileSync(datasetPath, TOML.stringify(datasetConfig), 'utf8');
                        console.log("Successfully rebuilt dataset.toml after auto split timesteps.");
                    }
                } catch (err) {
                    console.error("Failed to rebuild dataset.toml after auto split:", err);
                }
                fs.writeFileSync(progressPath, JSON.stringify({ current: 100, total: 100, status: 'done' }), 'utf8');
            } else {
                fs.writeFileSync(progressPath, JSON.stringify({ current: 0, total: 100, status: `error: exited with code ${code}` }), 'utf8');
            }
        });

        res.json({ success: true, message: 'Classify task started' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/jobs/:name/datasets/split-timesteps/progress', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const progressPath = path.join(jobPath, 'split_progress.json');
        if (fs.existsSync(progressPath)) {
            const data = JSON.parse(fs.readFileSync(progressPath, 'utf8'));
            return res.json(data);
        }
        res.json({ current: 0, total: 100, status: 'Not started' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/tag-captions', async (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (!fs.existsSync(jobPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const requestedImageDir = stripQuotes(String(req.body.image_dir || '').trim());
        const caption_extension = normalizeCaptionExtension(req.body.caption_extension || '.txt');
        const include_char = req.body.include_char !== false;
        const include_rating = req.body.include_rating !== false;
        const include_general = req.body.include_general !== false;
        const repoId = String(req.body.repo_id || 'Makki2104/animetimm/eva02_large_patch14_448.dbv4-full').trim();

        const targets = resolveTaggerTargetsForJob(jobPath, requestedImageDir);
        if (requestedImageDir) {
            const nativeRequestedDir = toNativePath(requestedImageDir);
            if (!fs.existsSync(nativeRequestedDir) || !fs.statSync(nativeRequestedDir).isDirectory()) {
                return res.status(400).json({ error: `Image directory not found: ${nativeRequestedDir}` });
            }
        }
        if (targets.imagePaths.length < 1) {
            return res.status(400).json({ error: 'No sd-scripts matched images found for tagging' });
        }
        if (!/^\.[A-Za-z0-9_-]+$/.test(caption_extension)) {
            return res.status(400).json({ error: 'Caption extension must look like .txt' });
        }

        const globalConfig = getGlobalConfig();
        const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
        const venv = getVenvPaths(venvPath);
        const taggerScript = path.join(ROOT_DIR, 'tools/tag_images_by_multilabel_timm.py');

        if (!fs.existsSync(venv.python)) {
            return res.status(500).json({ error: `Python not found in venv: ${venv.python}` });
        }
        if (!fs.existsSync(taggerScript)) {
            return res.status(500).json({ error: `Tagger script not found: ${taggerScript}` });
        }

        const taggerLogsDir = path.join(jobPath, 'logs');
        fs.mkdirSync(taggerLogsDir, { recursive: true });
        const imageListPath = path.join(taggerLogsDir, `tagger_images_${Date.now()}.txt`);
        fs.writeFileSync(imageListPath, targets.imagePaths.join('\n'), 'utf8');

        const args = [
            taggerScript,
            '--image-dir', targets.folders[0]?.nativeImageDir || jobPath,
            '--image-list', imageListPath,
            '--caption-extension', caption_extension,
            '--repo-id', repoId,
        ];
        if (!include_char) args.push('--no-include-char');
        if (!include_rating) args.push('--no-include-rating');
        if (!include_general) args.push('--no-include-general');

        const child = spawn(venv.python, args, {
            cwd: ROOT_DIR,
            env: {
                ...process.env,
                PYTHONIOENCODING: 'utf-8',
                PYTHONUTF8: '1',
            },
            windowsHide: true,
        });

        res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache');

        let stdoutBuffer = '';
        let stderr = '';
        let hasDone = false;
        const writeEvent = event => {
            if (!res.writableEnded) {
                res.write(`${JSON.stringify({
                    ...event,
                    matched_folder_count: targets.matchedFolderCount,
                    matched_folders: summarizeMatchedFolders(targets.folders)
                })}\n`);
            }
        };
        child.stdout.on('data', chunk => {
            stdoutBuffer += chunk.toString('utf8');
            const lines = stdoutBuffer.split(/\r?\n/);
            stdoutBuffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const event = JSON.parse(line);
                    if (event.type === 'done') hasDone = true;
                    writeEvent(event);
                } catch (_) {
                    writeEvent({ type: 'log', message: line });
                }
            }
        });
        child.stderr.on('data', chunk => { stderr += chunk.toString('utf8'); });
        child.on('error', err => {
            writeEvent({ type: 'error', error: err.message });
            if (!res.writableEnded) res.end();
        });
        child.on('close', code => {
            fs.promises.unlink(imageListPath).catch(() => {});
            if (stdoutBuffer.trim()) {
                try {
                    const event = JSON.parse(stdoutBuffer.trim());
                    if (event.type === 'done') hasDone = true;
                    writeEvent(event);
                } catch (_) { }
            }
            if (code !== 0 && !hasDone) {
                writeEvent({ type: 'error', error: stderr.trim() || `Tagger exited with code ${code}` });
            }
            if (!res.writableEnded) res.end();
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Delete job
app.delete('/api/jobs/:name', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (runningJobs.has(req.params.name)) {
            return res.status(400).json({ error: 'Stop job before deleting' });
        }
        if (fs.existsSync(jobPath)) {
            fs.rmSync(jobPath, { recursive: true });
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Clone job
app.post('/api/jobs/:name/clone', (req, res) => {
    try {
        const sourceName = sanitizeName(req.params.name);
        const sourcePath = getJobPath(sourceName);

        if (!fs.existsSync(sourcePath)) {
            return res.status(404).json({ error: 'Source job not found' });
        }

        let targetName = req.body.newName ? sanitizeName(req.body.newName) : null;

        // Auto-generate name if not provided
        if (!targetName) {
            targetName = `${sourceName}_copy`;
            let counter = 1;
            while (fs.existsSync(getJobPath(targetName))) {
                counter++;
                targetName = `${sourceName}_copy_${counter}`;
            }
        }

        const targetPath = getJobPath(targetName);
        if (fs.existsSync(targetPath)) {
            return res.status(409).json({ error: `Job "${targetName}" already exists` });
        }

        fs.mkdirSync(targetPath, { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'output'), { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'logs'), { recursive: true });
        fs.mkdirSync(path.join(targetPath, 'samples'), { recursive: true });

        // Copy config files
        ['dataset.toml', 'sample_prompts.txt'].forEach(file => {
            const src = path.join(sourcePath, file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, path.join(targetPath, file));
            }
        });

        // Handle config.toml special case: update output_name
        const configSrc = path.join(sourcePath, 'config.toml');
        if (fs.existsSync(configSrc)) {
            let config = TOML.parse(fs.readFileSync(configSrc, 'utf8'));

            // Sync output_name with new job name
            if (!config.training_arguments) config.training_arguments = {};
            config.training_arguments.output_name = targetName;

            fs.writeFileSync(path.join(targetPath, 'config.toml'), TOML.stringify(config), 'utf8');
        }

        res.json({ success: true, name: targetName });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Prompts API ---

app.get('/api/jobs/:name/prompts', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');
        if (!fs.existsSync(promptsPath)) {
            return res.json({ prompts: [] });
        }
        const text = fs.readFileSync(promptsPath, 'utf8').trim();
        const prompts = text ? text.split('\n').map(l => l.trim()).filter(l => l.length > 0) : [];
        res.json({ prompts });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.put('/api/jobs/:name/prompts', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');
        const prompts = req.body.prompts || [];
        fs.writeFileSync(promptsPath, prompts.join('\n') + (prompts.length ? '\n' : ''), 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

let persistentGenProcess = null; // { process, port, jobName }
const GEN_SERVER_PORT = 5000; // Fixed port for now

// Kill all running jobs when the Node server itself exits
function killAllJobs() {
    queueCoordinator.cancel();
    if (queueReconcileTimer) clearInterval(queueReconcileTimer);
    for (const [, job] of runningJobs) {
        if (job.pid) {
            try {
                if (process.platform === 'win32') {
                    execFileSync('taskkill', ['/PID', String(job.pid), '/F', '/T'], {
                        stdio: 'ignore',
                        windowsHide: true
                    });
                } else {
                    try { process.kill(-job.pid, 'SIGKILL'); } catch (_) {
                        try { process.kill(job.pid, 'SIGKILL'); } catch (__) {}
                    }
                }
            } catch (_) {}
        }
    }
    if (persistentGenProcess && persistentGenProcess.process) {
        const pid = persistentGenProcess.process.pid;
        try {
            if (process.platform === 'win32') {
                execFileSync('taskkill', ['/PID', String(pid), '/F', '/T'], {
                    stdio: 'ignore',
                    windowsHide: true
                });
            } else {
                try { process.kill(-pid, 'SIGKILL'); } catch (_) {
                    try { process.kill(pid, 'SIGKILL'); } catch (__) {}
                }
            }
        } catch (_) {}
    }
    serverInstanceHandle?.release();
}

for (const sig of ['exit', 'SIGINT', 'SIGTERM']) {
    process.on(sig, () => {
        killAllJobs();
        if (sig !== 'exit') process.exit(0);
    });
}

// Cross-platform process killer.
function killProcess(pid, gracefulMs = 8000) {
    return new Promise((resolve) => {
        if (process.platform === 'win32') {
            // taskkill /T kills the entire process tree on Windows.
            const k = spawn('taskkill', ['/PID', pid.toString(), '/F', '/T']);
            k.on('close', () => resolve());
            k.on('error', () => resolve());
            return;
        }

        // Linux/Mac: kill the entire process group
        const groupKill = (sig) => {
            try { process.kill(-pid, sig); } catch (_) {
                try { process.kill(pid, sig); } catch (__) {}
            }
        };

        if (gracefulMs <= 0) {
            groupKill('SIGKILL');
            resolve();
            return;
        }

        // SIGTERM -> give the training script a chance to flush the last checkpoint
        groupKill('SIGTERM');

        const timer = setTimeout(() => {
            groupKill('SIGKILL');
            resolve();
        }, gracefulMs);

        const poll = setInterval(() => {
            try {
                process.kill(pid, 0); // throws if pid is gone
            } catch (_) {
                clearInterval(poll);
                clearTimeout(timer);
                resolve();
            }
        }, 200);
    });
}

// --- Cross-platform venv/spawn helpers ---

// Open a file path or URL in the system file manager / browser
function openNative(target, isUrl = false) {
    if (isWindows) {
        spawn('explorer', [target]);
    } else if (isWSL) {
        let winTarget;
        if (isUrl) {
            winTarget = target;
        } else if (/^[A-Za-z]:[\\\/]/.test(target)) {
            // Already a Windows-style path (e.g. C:\foo), pass directly to explorer
            winTarget = target;
        } else {
            // Linux path -> convert to Windows UNC path via wslpath
            winTarget = require('child_process').execSync(`wslpath -w "${target}"`).toString().trim();
        }
        spawn('explorer.exe', [winTarget]);
    } else if (process.platform === 'darwin') {
        spawn('open', [target]);
    } else {
        spawn('xdg-open', [target]);
    }
}

// Convert a Windows-style path to a WSL /mnt/... path when running under WSL
function toNativePath(p) {
    if (!isWSL || typeof p !== 'string' || p.trim() === '') return p;
    return p.replace(/^([A-Za-z]):[\\\/]/, (_, d) => `/mnt/${d.toLowerCase()}/`)
             .replace(/\\/g, '/');
}

// Recursively convert all string values in an object/array
function convertPathsInObject(obj) {
    if (typeof obj === 'string') return toNativePath(obj);
    if (Array.isArray(obj)) return obj.map(convertPathsInObject);
    if (obj && typeof obj === 'object') {
        const out = {};
        for (const k of Object.keys(obj)) out[k] = convertPathsInObject(obj[k]);
        return out;
    }
    return obj;
}

function getVenvPaths(venvPath) {
    if (isWindows) {
        return {
            activate: path.join(venvPath, 'Scripts', 'Activate.ps1'),
            accelerate: path.join(venvPath, 'Scripts', 'accelerate.exe'),
            python: path.join(venvPath, 'Scripts', 'python.exe'),
        };
    } else {
        return {
            activate: path.join(venvPath, 'bin', 'activate'),
            accelerate: path.join(venvPath, 'bin', 'accelerate'),
            python: path.join(venvPath, 'bin', 'python'),
        };
    }
}

function buildEnvVar(name, value) {
    return isWindows ? `$env:${name}='${value}';` : `export ${name}='${value}';`;
}

// Returns { gpuEnv, accelerateFlags } or { error }
function buildLaunchConfig(gpuIds, mergedConfig, mergedConfigPath, jobArch) {
    const ta = mergedConfig.training_arguments || {};
    const mixedPrec = ta.mixed_precision || 'bf16';

    let gpuEnv = '';
    let accelerateFlags = `--mixed_precision ${mixedPrec}`;

    if (gpuIds) {
        if (!/^[\d\s,]+$/.test(gpuIds))
            return { error: `Invalid GPU IDs format: "${gpuIds}". Use numbers separated by commas (e.g. "0,1").` };

        const validIds = gpuIds.split(',').map(s => s.trim()).filter(Boolean);
        if (validIds.some(id => isNaN(parseInt(id))))
            return { error: 'GPU IDs must be valid numbers.' };
        if (validIds.length > 1)
            return { error: 'The sd-scripts NEW backend is wired for single-process training here. Select one GPU for training.' };

        gpuEnv = buildEnvVar('CUDA_VISIBLE_DEVICES', validIds.join(','));
    }

    if (ta.torch_compile)
        accelerateFlags += ' --dynamo_backend inductor';

    return { gpuEnv, accelerateFlags };
}

function buildTrainingLaunchCommand(jobName, jobPath, mergedConfig, mergedConfigPath) {
    const configPath = path.join(jobPath, 'config.toml');
    const jobConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));
    const globalConfig = getGlobalConfig();
    const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
    const venv = getVenvPaths(venvPath);
    const jobArch = getArchForJob(mergedConfig);
    const hasNetwork = !!(mergedConfig.network_arguments && mergedConfig.network_arguments.network_module);

    let currentGpuIds = '';
    try {
        currentGpuIds = jobConfig.gpu_ids ? jobConfig.gpu_ids.toString().trim() : '';
    } catch (err) {
        console.warn("Failed to parse config for GPU options:", err);
    }

    const launch = buildLaunchConfig(currentGpuIds, mergedConfig, mergedConfigPath, jobArch);
    if (launch.error) return { error: launch.error };
    const { gpuEnv, accelerateFlags } = launch;

    const scriptName = hasNetwork ? jobArch.scripts.train_network : jobArch.scripts.train;
    const targetScript = path.join(ROOT_DIR, scriptName);
    const baseTrainCmd = `python -m accelerate.commands.launch --num_cpu_threads_per_process 1 ${accelerateFlags} "${targetScript}" --config_file="${mergedConfigPath}"`;
    const customCliArgs = normalizeCustomCliArgs(jobConfig.ui_arguments?.custom_cli_args);
    const trainCmd = customCliArgs ? `${baseTrainCmd} ${customCliArgs}` : baseTrainCmd;

    const isMultiGpu = currentGpuIds && currentGpuIds.split(',').map(s => s.trim()).filter(s => s.length > 0).length > 1;
    const trainEnvVars = [
        buildEnvVar('PYTHONIOENCODING', 'utf-8'),
        buildEnvVar('TOKENIZERS_PARALLELISM', 'false'),
        buildEnvVar('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'),
        gpuEnv,
        mergedConfig.training_arguments?.step_profile ? buildEnvVar('STEP_PROFILE', '1') : '',
        mergedConfig.training_arguments?.profile_microbatch ? buildEnvVar('PROFILE_MICROBATCH', '1') : '',
        (isWindows && isMultiGpu) ? buildEnvVar('USE_LIBUV', '0') : '',
        (isWindows && isMultiGpu) ? buildEnvVar('MASTER_ADDR', '127.0.0.1') : '',
        (isWindows && isMultiGpu) ? buildEnvVar('MASTER_PORT', '29500') : ''
    ].filter(Boolean).join('\n');

    return {
        baseTrainCmd,
        trainCmd,
        baseTrainScript: buildShellScript(venv.activate, trainEnvVars, baseTrainCmd),
        trainScript: buildShellScript(venv.activate, trainEnvVars, trainCmd),
        customCliArgs
    };
}

function buildShellScript(activatePath, envVars, command) {
    if (isWindows) {
        return `& "${activatePath}";\n${envVars}\n${command}`;
    } else {
        return `source "${activatePath}"\n${envVars}\n${command}`;
    }
}

function spawnShell(script, cwd) {
    if (isWindows) {
        return spawn('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], {
            cwd,
            stdio: ['pipe', 'pipe', 'pipe']
        });
    } else {
        return spawn('bash', ['-c', script], {
            cwd,
            stdio: ['pipe', 'pipe', 'pipe'],
            detached: true
        });
    }
}

function killPersistentGen() {
    if (persistentGenProcess) {
        console.log(`Stop persistent gen server (PID: ${persistentGenProcess.process.pid})`);
        try {
            // Try graceful stop via API first
            fetch(`http://localhost:${persistentGenProcess.port}/stop`, { method: 'POST' }).catch(() => { });

            // Force kill after short delay
            setTimeout(() => {
                if (persistentGenProcess && persistentGenProcess.process) {
                    killProcess(persistentGenProcess.process.pid);
                    persistentGenProcess = null;
                }
            }, 1000);
        } catch (e) {
            persistentGenProcess = null;
        }
    }
}

app.post('/api/jobs/:name/unload', (req, res) => {
    try {
        if (persistentGenProcess) {
            killPersistentGen();
            res.json({ success: true, message: "Model unloaded" });
        } else {
            res.json({ success: true, message: "No model loaded" });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/train/stop', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const job = stopTrainingJob(jobName);

        if (!job) {
            return res.status(400).json({ error: 'Job not running' });
        }

        setQueueAutoRunning(false);
        queueCoordinator.cancel();
        trainingQueue.clearActive(jobName);
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/tensorboard/stop', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const tb = tbProcesses.get(jobName);

        if (!tb) {
            return res.json({ success: true, message: 'Not running' });
        }

        if (tb.pid) {
            await killProcess(tb.pid);
        }

        tbProcesses.delete(jobName);
        res.json({ success: true });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


app.get('/api/jobs/:name/checkpoints', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const outputDir = path.join(jobPath, 'output');
        if (!fs.existsSync(outputDir)) return res.json([]);

        const files = fs.readdirSync(outputDir)
            .filter(f => f.endsWith('.safetensors'))
            .map(f => {
                const stat = fs.statSync(path.join(outputDir, f));
                return { name: f, path: path.join(outputDir, f), mtime: stat.mtimeMs };
            })
            .sort((a, b) => b.mtime - a.mtime);

        res.json(files);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/generate', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        if (runningJobs.has(jobName)) {
            return res.status(400).json({ error: 'Job is running. Stop it first.' });
        }

        const jobPath = getJobPath(jobName);
        const configPath = path.join(jobPath, 'config.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }

        // Merged config for paths/args
        const mergedConfig = buildTrainingConfig(jobName, jobPath);

        const outputDir = path.join(jobPath, 'output');
        const promptsPath = path.join(jobPath, 'sample_prompts.txt');

        if (!fs.existsSync(promptsPath) || fs.readFileSync(promptsPath, 'utf8').trim().length === 0) {
            return res.status(400).json({ error: 'No sample prompts found. Add prompts in the Prompts tab.' });
        }

        const globalConfig = getGlobalConfig();
        const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
        const venv = getVenvPaths(venvPath);

        // Resolve architecture from job config
        const genArch = getArchForJob(mergedConfig);
        const genScript = path.join(ROOT_DIR, genArch.scripts.generate);

        // Extract args
        const mArgs = mergedConfig.model_arguments;
        const tArgs = mergedConfig.training_arguments;
        const archSection = mergedConfig[genArch.training_section] || {};

        // Read config to check for GPU IDs - prefer gen-specific GPU selection
        let gpuEnv = '';
        const genGpuIds = req.body.gen_gpu_ids || '';
        const rawConfig = TOML.parse(fs.readFileSync(configPath, 'utf8'));
        const configGpuIds = rawConfig.gpu_ids ? rawConfig.gpu_ids.toString().trim() : '';

        const currentGpuIdsRaw = genGpuIds || configGpuIds;
        const genGpuIdsNormalized = currentGpuIdsRaw.split(',').map(s => s.trim()).filter(s => s.length > 0).sort().join(',');

        if (genGpuIdsNormalized) {
            if (/^[\d\s,]+$/.test(genGpuIdsNormalized)) {
                gpuEnv = buildEnvVar('CUDA_VISIBLE_DEVICES', genGpuIdsNormalized);
                console.log(`[Gen] Using GPU isolation: ${gpuEnv}`);
            }
        }

        const genGpuCount = genGpuIdsNormalized ? genGpuIdsNormalized.split(',').length : 0;
        if (genGpuCount > 1) {
            return res.status(400).json({
                error: 'The sd-scripts NEW backend is wired for single-process generation here. Select one GPU for generation.'
            });
        }

        // Build model path args from registry. Training and inference use different flag names.
        const args = [];
        const missingPaths = [];
        for (const [configKey, pathDef] of Object.entries(genArch.global_paths)) {
            const val = mArgs[pathDef.cli_flag] || '';
            if (!val) {
                missingPaths.push(configKey);
                continue;
            }
            const genFlag = pathDef.gen_flag || pathDef.cli_flag;
            args.push(`--${genFlag}="${val}"`);
        }
        if (missingPaths.length > 0) {
            return res.status(400).json({
                error: `Missing model path(s) in Global Settings: ${missingPaths.join(', ')}`
            });
        }

        // anima_minimal_inference.py consumes the same prompt-file syntax used by the UI.
        args.push(
            `--from_file="${promptsPath}"`,
            `--save_path="${outputDir}"`,
            '--output_type=images',
            `--seed=${tArgs.seed ?? 42}`
        );

        // Add architecture-specific gen params from registry defaults + job overrides
        for (const [paramKey, paramDef] of Object.entries(genArch.gen_params || {})) {
            const val = req.body[paramKey] ?? archSection[paramKey] ?? paramDef.default;
            if (paramDef.type === 'text') {
                if (val) args.push(`--${paramDef.cli_flag}="${val}"`);
            } else {
                args.push(`--${paramDef.cli_flag}=${val}`);
            }
        }

        // Attention support for anima_minimal_inference.py
        if (req.body.flash_attn) {
            args.push('--attn_mode=flash');
        } else if (req.body.sage_attn) {
            args.push('--attn_mode=sageattn');
        }

        if (genArch.id === 'krea2') {
            const blocksToSwap = Number(req.body.blocks_to_swap ?? tArgs.blocks_to_swap ?? 0);
            if (Number.isInteger(blocksToSwap) && blocksToSwap > 0) {
                args.push(`--blocks_to_swap=${blocksToSwap}`);
            }

            const networkModule = mergedConfig.network_arguments?.network_module || 'networks.lora_krea2';
            const postHocMergeModules = new Set(['networks.cdka', 'networks.krona']);
            const hasPostHocAdapter = Boolean(req.body.network_weights) && postHocMergeModules.has(networkModule);
            if (archSection.fp8_scaled && !hasPostHocAdapter) {
                args.push('--fp8_scaled');
            }
        }

        // LoRA support
        if (req.body.network_weights) {
            const nw = stripQuotes(req.body.network_weights);
            if (mergedConfig.network_arguments?.network_module) {
                args.push(`--network_module=${mergedConfig.network_arguments.network_module}`);
            }
            args.push(`--lora_weight="${nw}"`);
            args.push(`--lora_multiplier=${req.body.network_mul || 1.0}`);
        }

        // Ensure logs dir exists
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

        if (persistentGenProcess) {
            killPersistentGen();
        }

        const oneShotEnvVars = [
            buildEnvVar('PYTHONIOENCODING', 'utf-8'),
            gpuEnv
        ].filter(Boolean).join('\n');
        const oneShotCmd = `python "${genScript}" ${args.join(' ')}`;
        const oneShotScript = buildShellScript(venv.activate, oneShotEnvVars, oneShotCmd);

        const oneShotProc = spawnShell(oneShotScript, ROOT_DIR);

        const oneShotLogFileName = `gen_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
        const oneShotLogStream = fs.createWriteStream(path.join(logsDir, oneShotLogFileName), { flags: 'a' });

        const oneShotAppendLog = (data) => {
            const text = data.toString();
            oneShotLogStream.write(text);
            broadcastLog(jobName, text);
        };

        oneShotProc.stdout.on('data', oneShotAppendLog);
        oneShotProc.stderr.on('data', oneShotAppendLog);

        oneShotProc.stdout.on('error', (err) => console.error(`[Gen/stdout] ${err.message}`));
        oneShotProc.stderr.on('error', (err) => console.error(`[Gen/stderr] ${err.message}`));
        oneShotLogStream.on('error', (err) => console.error(`[Gen/LogFile] ${err.message}`));

        oneShotProc.on('close', (code) => {
            const msg = `\n--- Generation finished (exit code: ${code}) ---\n`;
            oneShotLogStream.write(msg);
            oneShotLogStream.end();
            broadcastLog(jobName, msg);
            runningJobs.delete(jobName);
            broadcastStatus(jobName, 'idle');
        });

        runningJobs.set(jobName, {
            process: oneShotProc,
            pid: oneShotProc.pid,
            startTime: Date.now(),
            type: 'generation',
            gpuIds: currentGpuIdsRaw
        });

        broadcastStatus(jobName, 'generating');
        res.json({ success: true, pid: oneShotProc.pid });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Training Control ---

async function startQueuedJob(jobName) {
    recordQueueEvent('start-requested', { jobName, port: activePort });
    const response = await fetch(`http://127.0.0.1:${activePort}/api/jobs/${encodeURIComponent(jobName)}/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fromQueue: true })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || body.error) {
        const error = body.error || `Failed to start queued job "${jobName}"`;
        setQueueAutoRunning(false, error);
        trainingQueue.clearActive(jobName);
        broadcastStatus(jobName, 'failed');
        recordQueueEvent('start-failed', { jobName, status: response.status, error });
        throw new Error(error);
    }
    recordQueueEvent('start-succeeded', { jobName, pid: body.pid });
    return body;
}

async function runNextQueuedJob() {
    const outcome = await queueCoordinator.advanceNow();
    return outcome.started ? outcome.result : null;
}

app.get('/api/queue', (req, res) => {
    try {
        res.json({ ...trainingQueue.getState(), autoRunning: queueAutoRunning });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/queue/jobs/:name', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        res.json({ ...trainingQueue.enqueue(jobName), autoRunning: queueAutoRunning });
    } catch (err) {
        res.status(err.message === 'Job not found' ? 404 : 500).json({ error: err.message });
    }
});

app.delete('/api/queue/jobs/:name', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        if (runningJobs.has(jobName)) {
            return res.status(400).json({ error: 'Stop job before removing it from queue' });
        }
        res.json({ ...trainingQueue.remove(jobName), autoRunning: queueAutoRunning });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/queue/jobs/:name/move', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const targetIndex = Number.parseInt(req.body?.index, 10);
        if (!Number.isFinite(targetIndex)) {
            return res.status(400).json({ error: 'index required' });
        }
        res.json({ ...trainingQueue.move(jobName, targetIndex), autoRunning: queueAutoRunning });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/queue/start', async (req, res) => {
    try {
        const runningTraining = await getRunningTrainingJobNameFresh();
        const state = trainingQueue.getState();
        if (state.active && !runningJobs.has(state.active)) {
            trainingQueue.clearActive(state.active);
        }

        setQueueAutoRunning(true);
        recordQueueEvent('queue-armed', { runningTraining, items: trainingQueue.getState().items });
        if (runningTraining) {
            queueCoordinator.requestAdvance(2000);
            return res.json({
                success: true,
                started: false,
                waitingFor: runningTraining,
                queue: trainingQueue.getState(),
                autoRunning: queueAutoRunning
            });
        }

        const result = await runNextQueuedJob();
        res.json({
            success: true,
            started: !!result,
            queue: trainingQueue.getState(),
            autoRunning: queueAutoRunning
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/queue/stop', async (req, res) => {
    try {
        setQueueAutoRunning(false);
        queueCoordinator.cancel();
        recordQueueEvent('queue-paused');
        const stoppedJob = await stopRunningTrainingForQueue();
        res.json({
            success: true,
            stoppedJob,
            queue: trainingQueue.getState(),
            autoRunning: queueAutoRunning
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/train/start', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const fromQueue = req.body?.fromQueue === true;
        const jobPath = getJobPath(jobName);
        const configPath = path.join(jobPath, 'config.toml');

        if (!fs.existsSync(configPath)) {
            return res.status(404).json({ error: 'Job not found' });
        }
        const runningTraining = await getRunningTrainingJobNameFresh();
        if (runningTraining === jobName || await isJobTrainingFresh(jobName)) {
            return res.status(400).json({ error: 'Job already running' });
        }
        if (runningTraining) {
            return res.status(400).json({ error: `Another training job is already running: ${runningTraining}` });
        }
        if (fromQueue && !trainingQueue.getState().items.includes(jobName)) {
            trainingQueue.enqueue(jobName);
        }

        // Auto-kill persistent gen server to free VRAM
        if (persistentGenProcess) {
            console.log("Stopping persistent generation server before training...");
            killPersistentGen();
        }

        // Build merged config and write to temp file
        const mergedConfig = buildTrainingConfig(jobName, jobPath);

        // Convert Windows paths to WSL paths when running under WSL
        if (isWSL) {
            // Convert all paths in merged config (model paths, output dirs, etc.)
            const converted = convertPathsInObject(mergedConfig);
            // Also convert image_dir entries inside dataset.toml -> write a WSL version
            const datasetPath = path.join(jobPath, 'dataset.toml');
            if (fs.existsSync(datasetPath)) {
                const datasetRaw = TOML.parse(fs.readFileSync(datasetPath, 'utf8'));
                const datasetConverted = convertPathsInObject(datasetRaw);
                const mergedDatasetPath = path.join(jobPath, '_merged_dataset.toml');
                fs.writeFileSync(mergedDatasetPath, TOML.stringify(datasetConverted), 'utf8');
                if (converted.dataset_arguments) {
                    converted.dataset_arguments.dataset_config = mergedDatasetPath;
                }
            }
            Object.assign(mergedConfig, converted);
        }

        const mergedConfigPath = path.join(jobPath, '_merged_config.toml');
        fs.writeFileSync(mergedConfigPath, TOML.stringify(mergedConfig), 'utf8');

        // Ensure output dirs exist
        const outputDir = path.join(jobPath, 'output');
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
        if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

        // A completion signal belongs to one training run. Remove a stale signal
        // before starting so a previous server crash cannot mark this run as done.
        const completionSignalPath = path.join(outputDir, 'completed.signal');
        try {
            fs.rmSync(completionSignalPath, { force: true });
        } catch (err) {
            return res.status(500).json({ error: `Failed to clear completion signal: ${err.message}` });
        }

        const launch = buildTrainingLaunchCommand(jobName, jobPath, mergedConfig, mergedConfigPath);
        if (launch.error) return res.status(400).json({ error: launch.error });
        const { trainScript } = launch;

        const scriptPath = path.join(jobPath, isWindows ? 'launch_command.ps1' : 'launch_command.sh');
        fs.writeFileSync(scriptPath, trainScript, 'utf8');
        console.log(`\n--- Training Launch Command for ${jobName} ---`);
        console.log(trainScript);
        console.log("----------------------------------------------\n");

        const proc = spawnShell(trainScript, ROOT_DIR);

        const logBuffer = [];
        const MAX_LOG_LINES = 5000;

        // Write logs to file
        const logFileName = `train_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
        const logStream = fs.createWriteStream(path.join(logsDir, logFileName), { flags: 'a' });

        const appendLog = (data, cliStream = null) => {
            const text = data.toString();
            logBuffer.push(text);
            if (logBuffer.length > MAX_LOG_LINES) logBuffer.shift();
            if (cliStream) cliStream.write(text);
            logStream.write(text);
            broadcastLog(jobName, text);
        };

        proc.stdout.on('data', (data) => appendLog(data, process.stdout));
        proc.stderr.on('data', (data) => appendLog(data, process.stderr));

        // Prevent crashes on stream errors
        proc.stdout.on('error', (err) => console.error(`[Train/stdout] ${err.message}`));
        proc.stderr.on('error', (err) => console.error(`[Train/stderr] ${err.message}`));
        logStream.on('error', (err) => console.error(`[Train/LogFile] ${err.message}`));

        proc.on('close', (code) => {
            const stoppedByRequest = runningJobs.get(jobName)?.stopRequested === true;

            // The signal file is the strongest completion indication, while the
            // exit/log fallback handles wrappers such as Accelerate/PowerShell
            // that can return a non-zero code after a successful final save.
            const hasCompletionSignal = fs.existsSync(completionSignalPath);
            const completedSuccessfully = !stoppedByRequest && (
                hasCompletionSignal || isSuccessfulTrainingExit({
                    code,
                    stoppedByRequest,
                    logText: logBuffer.join('')
                })
            );
            if (hasCompletionSignal) {
                try {
                    fs.unlinkSync(completionSignalPath);
                } catch (e) {
                    console.error(`[Queue] Failed to delete completion signal file: ${e.message}`);
                }
            }

            const msg = `\n--- Training ${completedSuccessfully ? 'completed' : 'stopped'} (exit code: ${code}) ---\n`;
            appendLog(Buffer.from(msg));
            logStream.end();
            runningJobs.delete(jobName);
            clearTrainingProcessState(jobName);
            invalidateDetectedTrainingProcesses();
            const isQueuedJob = trainingQueue.getState().items.includes(jobName);
            if (fromQueue || isQueuedJob) {
                if (completedSuccessfully) {
                    trainingQueue.finishQueuedJob(jobName, { success: true });
                    broadcastStatus(jobName, 'completed');
                    recordQueueEvent('job-completed', {
                        jobName,
                        remaining: trainingQueue.getState().items,
                        autoRunning: queueAutoRunning
                    });
                    queueCoordinator.requestAdvance(1000);
                } else {
                    setQueueAutoRunning(false, `Training job "${jobName}" exited with code ${code}`);
                    queueCoordinator.cancel();
                    trainingQueue.finishQueuedJob(jobName, { success: false });
                    broadcastStatus(jobName, stoppedByRequest ? 'stopped' : 'failed');
                }
            } else {
                broadcastStatus(jobName, 'idle');
            }
        });

        proc.on('error', (err) => {
            appendLog(Buffer.from(`\nERROR: ${err.message}\n`));
            const stoppedByRequest = runningJobs.get(jobName)?.stopRequested === true;
            runningJobs.delete(jobName);
            if (!isPidAlive(proc.pid)) clearTrainingProcessState(jobName);
            invalidateDetectedTrainingProcesses();
            const isQueuedJob = trainingQueue.getState().items.includes(jobName);
            if (fromQueue || isQueuedJob) {
                setQueueAutoRunning(false, err.message);
                queueCoordinator.cancel();
                trainingQueue.finishQueuedJob(jobName, { success: false });
                broadcastStatus(jobName, stoppedByRequest ? 'stopped' : 'failed');
            } else {
                broadcastStatus(jobName, 'idle');
            }
        });

        const startTime = Date.now();
        runningJobs.set(jobName, {
            process: proc,
            pid: proc.pid,
            startTime,
            logBuffer,
            type: 'training',
            gpuIds: currentGpuIds,
            fromQueue
        });
        writeTrainingProcessState(jobName, {
            jobName,
            pid: proc.pid,
            startTime,
            type: 'training',
            gpuIds: currentGpuIds,
            fromQueue,
            mergedConfigPath
        });

        if (fromQueue) {
            trainingQueue.markActive(jobName);
        }
        broadcastStatus(jobName, 'running');
        res.json({ success: true, pid: proc.pid });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


app.get('/api/jobs/:name/train/status', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const isRunning = await isJobTrainingFresh(jobName);
        res.json({ running: isRunning });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- TensorBoard API ---

const tbProcesses = new Map(); // jobName -> { process, port }
let nextTbPort = 6006;

function getTensorBoardLogDir(jobPath) {
    const mergedConfigPath = path.join(jobPath, '_merged_config.toml');
    if (fs.existsSync(mergedConfigPath)) {
        try {
            const mergedConfig = TOML.parse(fs.readFileSync(mergedConfigPath, 'utf8'));
            const configuredDir = mergedConfig.training_arguments?.logging_dir;
            if (configuredDir && String(configuredDir).trim()) {
                return toNativePath(stripQuotes(String(configuredDir).trim()));
            }
        } catch (err) {
            console.warn(`[TensorBoard] Failed to read merged config logging_dir: ${err.message}`);
        }
    }

    return path.join(jobPath, 'output');
}

app.post('/api/jobs/:name/tensorboard', async (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const jobPath = getJobPath(jobName);
        const logsDir = getTensorBoardLogDir(jobPath);

        if (tbProcesses.has(jobName)) {
            const tb = tbProcesses.get(jobName);
            if (path.resolve(tb.logDir || '') === path.resolve(logsDir)) {
                return res.json({ success: true, port: tb.port, url: `http://localhost:${tb.port}`, logDir: tb.logDir });
            }
            if (tb.pid) {
                await killProcess(tb.pid).catch(() => {});
            }
            tbProcesses.delete(jobName);
        }

        if (!fs.existsSync(logsDir)) {
            fs.mkdirSync(logsDir, { recursive: true });
        }

        // Get venv path
        const globalConfig = getGlobalConfig();
        const venvPath = toNativePath(globalConfig.venv_path || path.join(ROOT_DIR, 'venv'));
        const venv = getVenvPaths(venvPath);

        const port = nextTbPort++;

        const tbCmd = `python -m tensorboard.main --logdir="${logsDir}" --port=${port} --host=0.0.0.0`;
        const tbScript = buildShellScript(venv.activate, '', tbCmd);
        const proc = spawnShell(tbScript, ROOT_DIR);

        proc.stderr.on('data', (data) => {
            const text = data.toString();
            console.log(`[TensorBoard/${jobName}] ${text.trim()}`);
        });

        console.log(`[TensorBoard/${jobName}] logdir=${logsDir}`);

        proc.on('close', (code) => {
            console.log(`[TensorBoard/${jobName}] Exited with code ${code}`);
            tbProcesses.delete(jobName);
        });

        tbProcesses.set(jobName, { process: proc, port, pid: proc.pid, logDir: logsDir });

        res.json({ success: true, port, url: `http://localhost:${port}`, logDir: logsDir });

    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});


app.get('/api/jobs/:name/tensorboard/status', (req, res) => {
    try {
        const jobName = sanitizeName(req.params.name);
        const tb = tbProcesses.get(jobName);
        if (tb) {
            res.json({ running: true, port: tb.port, url: `http://localhost:${tb.port}`, logDir: tb.logDir });
        } else {
            res.json({ running: false });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Samples API ---

function collectImages(dir, relBase, jobName) {
    const images = [];
    if (!fs.existsSync(dir)) return images;
    fs.readdirSync(dir, { withFileTypes: true }).forEach(entry => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            // Recurse into subdirectories (e.g. output/sample/)
            images.push(...collectImages(fullPath, path.join(relBase, entry.name), jobName));
        } else if (/\.(png|jpg|jpeg|webp)$/i.test(entry.name)) {
            const stat = fs.statSync(fullPath);
            const relPath = path.join(relBase, entry.name).replace(/\\/g, '/');
            images.push({
                name: entry.name,
                dir: relBase.replace(/\\/g, '/'),
                mtime: stat.mtimeMs,
                path: `/api/jobs/${jobName}/samples/${relPath}`
            });
        }
    });
    return images;
}

app.get('/api/jobs/:name/samples', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const samplesDir = path.join(jobPath, 'samples');
        const outputDir = path.join(jobPath, 'output');

        let images = [];
        images.push(...collectImages(samplesDir, 'samples', req.params.name));
        images.push(...collectImages(outputDir, 'output', req.params.name));

        images.sort((a, b) => b.mtime - a.mtime);
        res.set('Cache-Control', 'no-store');
        res.json(images);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Serve sample image files (supports nested paths like output/sample/img.png)
app.get('/api/jobs/:name/samples/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0]; // everything after /samples/
        const filePath = path.join(jobPath, relativePath);
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.sendFile(filePath);
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Extract PNG Metadata helper
function extractPngMetadata(filePath) {
    try {
        const buffer = fs.readFileSync(filePath);
        let offset = 8; // skip PNG signature
        const metadata = {};

        while (offset < buffer.length) {
            const length = buffer.readUInt32BE(offset);
            const type = buffer.slice(offset + 4, offset + 8).toString('ascii');

            if (type === 'tEXt') {
                const data = buffer.slice(offset + 8, offset + 8 + length).toString('utf8');
                const nullIdx = data.indexOf('\u0000');
                if (nullIdx !== -1) {
                    const key = data.substring(0, nullIdx);
                    const val = data.substring(nullIdx + 1);
                    metadata[key] = val;
                }
            } else if (type === 'IEND') break;

            offset += 12 + length;
        }
        return metadata;
    } catch (e) {
        return null;
    }
}

// Get image metadata
app.get('/api/jobs/:name/metadata/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0];
        const filePath = path.join(jobPath, relativePath);

        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const metadata = extractPngMetadata(filePath);
            res.json(metadata || {});
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Delete a sample image
app.delete('/api/jobs/:name/samples/*', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const relativePath = req.params[0];
        const filePath = path.join(jobPath, relativePath);

        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            res.json({ success: true });
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Job Settings Actions ---

app.post('/api/jobs/:name/open-folder', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        if (fs.existsSync(jobPath)) openNative(jobPath);
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/system/open-folder', (req, res) => {
    try {
        const { path: folderPath } = req.body;
        // On WSL the frontend sends Windows-style paths; convert for fs.existsSync
        const nativePath = toNativePath(folderPath);
        if (nativePath && fs.existsSync(nativePath)) {
            openNative(folderPath);
            res.json({ success: true });
        } else {
            res.status(404).json({ error: 'Folder not found' });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/system/select-folder', async (req, res) => {
    try {
        const selectedPath = await selectFolderDialog(req.body?.initial_path || '');
        res.json({ path: selectedPath || '' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/system/inspect-image-folder', async (req, res) => {
    try {
        const { path: folderPath, caption_extension, batch_import, auto_balance_repeats } = req.body;
        res.json(await inspectDatasetImageFolders({
            imageDir: folderPath,
            captionExtension: caption_extension,
            batchImport: batch_import === true,
            autoBalanceRepeats: auto_balance_repeats === true,
            toNativePath: p => toNativePath(stripQuotes(String(p || '').trim()))
        }));
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/clear-logs', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const logsPath = path.join(jobPath, 'logs');
        if (fs.existsSync(logsPath)) {
            fs.readdirSync(logsPath).forEach(file => {
                const filePath = path.join(logsPath, file);
                try {
                    const stat = fs.statSync(filePath);
                    if (stat.isFile()) fs.unlinkSync(filePath);
                    else if (stat.isDirectory()) fs.rmSync(filePath, { recursive: true });
                } catch (e) { }
            });
        }
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/jobs/:name/reset-config', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const { config } = getDefaultConfig();
        fs.writeFileSync(path.join(jobPath, 'config.toml'), TOML.stringify(config), 'utf8');
        const dataset = getDefaultDataset();
        fs.writeFileSync(path.join(jobPath, 'dataset.toml'), TOML.stringify(dataset), 'utf8');
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// --- Hardware Monitor ---

function getActiveGpus() {
    const active = {};
    for (const [, job] of runningJobs.entries()) {
        if (job.gpuIds) {
            job.gpuIds.split(',').forEach(id => {
                const trimmed = id.trim();
                if (trimmed) active[trimmed] = job.type === 'generation' ? 'sampling' : 'training';
            });
        }
    }
    if (persistentGenProcess && persistentGenProcess.gpuIds) {
        persistentGenProcess.gpuIds.split(',').forEach(id => {
            const trimmed = id.trim();
            if (trimmed) active[trimmed] = 'sampling';
        });
    }
    return active;
}

require('./lib/hardware').startHardwareMonitor(wss, getActiveGpus);

// Prevent server crash on unhandled errors
process.on('uncaughtException', (err) => {
    console.error('CRITICAL ERROR (Uncaught Exception):', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('CRITICAL ERROR (Unhandled Rejection) at:', promise, 'reason:', reason);
});

// --- Start Server ---

function checkPort(port) {
    return new Promise((resolve) => {
        const tester = net.createServer();
        tester.once('error', () => resolve(false));
        tester.once('listening', () => {
            tester.close();
            resolve(true);
        });
        tester.listen(port);
    });
}

(async () => {
    if (!await checkPort(DEFAULT_PORT)) {
        serverInstanceHandle.release();
        console.error(`\nERROR: Port ${DEFAULT_PORT} is already in use. Refusing to start a second Training UI instance.`);
        process.exit(1);
    }

    activePort = DEFAULT_PORT;
    server.listen(DEFAULT_PORT, () => {
        console.log(`Anima Training UI running at http://localhost:${DEFAULT_PORT}`);
        recordQueueEvent('server-started', {
            pid: process.pid,
            port: DEFAULT_PORT,
            autoRunning: queueAutoRunning,
            items: trainingQueue.getState().items
        });
        startQueueReconciler();
        reconcileQueue('startup').catch(err => recordQueueEvent('reconcile-error', {
            reason: 'startup',
            error: err.message
        }));
        try {
            openNative(`http://localhost:${DEFAULT_PORT}`, true);
        } catch (e) {
            console.warn('Could not open browser automatically.');
        }
    });
})();

app.get('/api/jobs/:name/logs', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(logsDir)) return res.json([]);

        const files = fs.readdirSync(logsDir)
            .filter(f => f.endsWith('.log'))
            .map(f => {
                const stat = fs.statSync(path.join(logsDir, f));
                return { name: f, size: stat.size, mtime: stat.mtimeMs };
            })
            .sort((a, b) => b.mtime - a.mtime);

        res.json(files);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/jobs/:name/logs/latest', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const logsDir = path.join(jobPath, 'logs');
        if (!fs.existsSync(logsDir)) {
            return res.json({ name: null, content: "" });
        }

        const files = fs.readdirSync(logsDir)
            .filter(f => f.endsWith('.log'))
            .map(f => {
                const stat = fs.statSync(path.join(logsDir, f));
                return { name: f, mtime: stat.mtimeMs };
            })
            .sort((a, b) => b.mtime - a.mtime);

        if (files.length === 0) {
            return res.json({ name: null, content: "" });
        }

        const latestFile = files[0].name;
        const filePath = path.join(logsDir, latestFile);
        const content = fs.readFileSync(filePath, 'utf8');

        res.json({ name: latestFile, content });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/jobs/:name/logs/:filename', (req, res) => {
    try {
        const jobPath = getJobPath(req.params.name);
        const logsDir = path.join(jobPath, 'logs');
        const filename = path.basename(req.params.filename);
        const logFilePath = path.join(logsDir, filename);

        if (!fs.existsSync(logFilePath)) {
            return res.status(404).json({ error: 'Log file not found' });
        }

        const content = fs.readFileSync(logFilePath, 'utf8');
        res.json({ name: filename, content });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});
