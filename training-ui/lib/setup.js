const { execFileSync, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.join(__dirname, '..', '..');

function execWindowsPowerShellSync(script, options = {}) {
    return execFileSync('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], {
        ...options,
        stdio: options.stdio ?? 'pipe',
        windowsHide: options.windowsHide ?? true
    });
}

function getPythonCmd() {
    const venvPath = path.join(ROOT_DIR, 'venv');
    if (process.platform === 'win32') {
        const p = path.join(venvPath, 'Scripts', 'python.exe');
        return fs.existsSync(p) ? p : 'python';
    } else {
        const p = path.join(venvPath, 'bin', 'python');
        return fs.existsSync(p) ? p : 'python3';
    }
}

function canImport(pythonCmd, moduleName) {
    try {
        if (process.platform === 'win32') {
            execWindowsPowerShellSync(`& "${pythonCmd}" -c 'import ${moduleName}'`, { stdio: 'pipe' });
        } else {
            execSync(`"${pythonCmd}" -c "import ${moduleName}"`, { stdio: 'pipe' });
        }
        return true;
    } catch {
        return false;
    }
}

function pipInstall(pythonCmd, pkgPath) {
    if (process.platform === 'win32') {
        execWindowsPowerShellSync(`& "${pythonCmd}" -m pip install --no-deps -e "${pkgPath}"`, { stdio: 'pipe' });
    } else {
        execSync(`"${pythonCmd}" -m pip install --no-deps -e '${pkgPath}'`, { stdio: 'pipe' });
    }
}

function ensurePackage(pkgPath, moduleName, { label, unavailableMsg, fixHint }) {
    if (!fs.existsSync(pkgPath)) return;
    const pythonCmd = getPythonCmd();
    if (canImport(pythonCmd, moduleName)) return;
    console.log(`[setup] Installing ${label}...`);
    try {
        pipInstall(pythonCmd, pkgPath);
        console.log(`[setup] ${label} installed.\n`);
    } catch {
        console.warn(`[setup] Could not install ${label}. ${unavailableMsg}`);
        if (fixHint) console.warn(`[setup] To fix manually, run: ${fixHint}\n`);
    }
}

function ensureCudaDirectBackend() {
    if (process.platform !== 'win32') return;
    ensurePackage(
        path.join(ROOT_DIR, 'cuda_direct_pkg'),
        'cuda_direct_backend',
        {
            label: 'cuda_direct_backend',
            unavailableMsg: 'Multi-GPU cuda_direct will be unavailable.',
            fixHint: null,
        }
    );
}

function ensureWdParallel() {
    const pkgPath = path.join(ROOT_DIR, 'wd_parallel_pkg');
    ensurePackage(
        pkgPath,
        'wd_parallel',
        {
            label: 'wd_parallel',
            unavailableMsg: 'TP/SP training may be unavailable.',
            fixHint: `pip install -e "${pkgPath}"`,
        }
    );
}

function runSetup() {
    ensureCudaDirectBackend();
    ensureWdParallel();
}

module.exports = { runSetup };
