const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

function defaultIsPidAlive(pid) {
    const numericPid = Number.parseInt(pid, 10);
    if (!Number.isFinite(numericPid) || numericPid <= 0) return false;
    try {
        process.kill(numericPid, 0);
        return true;
    } catch (_) {
        return false;
    }
}

function readLock(lockPath) {
    try {
        return JSON.parse(fs.readFileSync(lockPath, 'utf8'));
    } catch (_) {
        return null;
    }
}

function acquireServerInstance({
    lockPath,
    port,
    pid = process.pid,
    isPidAlive = defaultIsPidAlive
}) {
    if (!lockPath) throw new Error('lockPath is required');

    fs.mkdirSync(path.dirname(lockPath), { recursive: true });
    const instanceId = crypto.randomUUID();
    const lock = {
        instanceId,
        pid,
        port,
        startedAt: new Date().toISOString()
    };

    for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
            const fd = fs.openSync(lockPath, 'wx');
            try {
                fs.writeFileSync(fd, JSON.stringify(lock, null, 2), 'utf8');
            } finally {
                fs.closeSync(fd);
            }

            let released = false;
            return {
                acquired: true,
                lock,
                release() {
                    if (released) return;
                    released = true;
                    const current = readLock(lockPath);
                    if (current?.instanceId === instanceId) {
                        fs.rmSync(lockPath, { force: true });
                    }
                }
            };
        } catch (err) {
            if (err.code !== 'EEXIST') throw err;
            const existing = readLock(lockPath);
            if (existing?.pid && isPidAlive(existing.pid)) {
                return { acquired: false, existing, release() {} };
            }
            fs.rmSync(lockPath, { force: true });
        }
    }

    return { acquired: false, existing: readLock(lockPath), release() {} };
}

module.exports = { acquireServerInstance };
