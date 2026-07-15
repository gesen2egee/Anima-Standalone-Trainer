const { execFileSync } = require('child_process');
const path = require('path');

function normalizePid(value) {
    const pid = Number.parseInt(value, 10);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
}

function isPidAlive(value) {
    const pid = normalizePid(value);
    if (!pid) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch (_) {
        return false;
    }
}

function waitForExit(pid, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    const signal = new Int32Array(new SharedArrayBuffer(4));
    while (isPidAlive(pid) && Date.now() < deadline) {
        Atomics.wait(signal, 0, 0, 50);
    }
    return !isPidAlive(pid);
}

function readProcessInfo(value, platform = process.platform) {
    const pid = normalizePid(value);
    if (!pid) return null;
    try {
        if (platform === 'win32') {
            const script = `$p=Get-CimInstance Win32_Process -Filter "ProcessId=${pid}"; if($p){@{name=$p.Name;commandLine=$p.CommandLine}|ConvertTo-Json -Compress}`;
            const raw = execFileSync('powershell.exe', ['-NoProfile', '-Command', script], {
                encoding: 'utf8',
                windowsHide: true,
                stdio: ['ignore', 'pipe', 'ignore']
            }).trim();
            return raw ? JSON.parse(raw) : null;
        }
        const raw = execFileSync('ps', ['-p', String(pid), '-o', 'comm=', '-o', 'args='], {
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        }).trim();
        if (!raw) return null;
        const separator = raw.search(/\s/);
        return separator < 0
            ? { name: raw, commandLine: raw }
            : { name: raw.slice(0, separator), commandLine: raw.slice(separator).trim() };
    } catch (_) {
        return null;
    }
}

function isExpectedNodeServer(pid, serverScriptPath, readInfo = readProcessInfo) {
    const info = readInfo(pid);
    if (!info) return false;
    const name = path.basename(String(info.name || '')).toLowerCase();
    const commandLine = String(info.commandLine || '').replace(/\\/g, '/').toLowerCase();
    const scriptName = path.basename(serverScriptPath).toLowerCase();
    return (name === 'node' || name === 'node.exe') && commandLine.includes(scriptName);
}

function terminateProcessTree(value, platform = process.platform) {
    const pid = normalizePid(value);
    if (!pid || pid === process.pid) return false;
    try {
        if (platform === 'win32') {
            execFileSync('taskkill', ['/PID', String(pid), '/F', '/T'], {
                windowsHide: true,
                stdio: 'ignore'
            });
        } else {
            try { process.kill(-pid, 'SIGTERM'); } catch (_) { process.kill(pid, 'SIGTERM'); }
        }
    } catch (_) {
        return !isPidAlive(pid);
    }
    if (waitForExit(pid)) return true;
    if (platform !== 'win32') {
        try { process.kill(-pid, 'SIGKILL'); } catch (_) {
            try { process.kill(pid, 'SIGKILL'); } catch (__) { }
        }
    }
    return waitForExit(pid, 1000);
}

function findPortOwnerPid(port, platform = process.platform) {
    const numericPort = Number.parseInt(port, 10);
    if (!Number.isFinite(numericPort) || numericPort <= 0) return null;
    try {
        if (platform === 'win32') {
            const script = `$c=Get-NetTCPConnection -State Listen -LocalPort ${numericPort} -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){$c.OwningProcess}`;
            return normalizePid(execFileSync('powershell.exe', ['-NoProfile', '-Command', script], {
                encoding: 'utf8',
                windowsHide: true,
                stdio: ['ignore', 'pipe', 'ignore']
            }).trim());
        }
        return normalizePid(execFileSync('lsof', ['-nP', `-iTCP:${numericPort}`, '-sTCP:LISTEN', '-t'], {
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        }).trim().split(/\s+/)[0]);
    } catch (_) {
        return null;
    }
}

function stopPreviousNodeOnPort({ port, serverScriptPath, findOwner = findPortOwnerPid, isExpected = isExpectedNodeServer, terminate = terminateProcessTree }) {
    const pid = findOwner(port);
    if (!pid || !isExpected(pid, serverScriptPath)) return { stopped: false, pid };
    return { stopped: terminate(pid), pid };
}

function monitorParentProcess(parentPid, onMissing, { intervalMs = 2000, isAlive = isPidAlive } = {}) {
    const pid = normalizePid(parentPid);
    if (!pid) return null;
    const timer = setInterval(() => {
        if (isAlive(pid)) return;
        clearInterval(timer);
        onMissing(pid);
    }, intervalMs);
    timer.unref?.();
    return timer;
}

module.exports = {
    findPortOwnerPid,
    isExpectedNodeServer,
    isPidAlive,
    monitorParentProcess,
    stopPreviousNodeOnPort,
    terminateProcessTree
};
