const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');

const { acquireServerInstance } = require('./serverInstance');

function makeLockPath() {
    return path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'anima-instance-')), 'server_instance.json');
}

test('only one live server instance can own the lock', () => {
    const lockPath = makeLockPath();
    const first = acquireServerInstance({ lockPath, port: 3000, pid: 101, isPidAlive: () => true });
    const second = acquireServerInstance({ lockPath, port: 3000, pid: 202, isPidAlive: () => true });

    assert.strictEqual(first.acquired, true);
    assert.strictEqual(second.acquired, false);
    assert.strictEqual(second.existing.pid, 101);

    first.release();
    assert.strictEqual(fs.existsSync(lockPath), false);
});

test('stale instance lock is replaced', () => {
    const lockPath = makeLockPath();
    fs.writeFileSync(lockPath, JSON.stringify({ instanceId: 'stale', pid: 101, port: 3000 }), 'utf8');

    const instance = acquireServerInstance({ lockPath, port: 3000, pid: 202, isPidAlive: () => false });

    assert.strictEqual(instance.acquired, true);
    assert.strictEqual(JSON.parse(fs.readFileSync(lockPath, 'utf8')).pid, 202);
    instance.release();
});

test('an old owner cannot remove a newer instance lock', () => {
    const lockPath = makeLockPath();
    const first = acquireServerInstance({ lockPath, port: 3000, pid: 101, isPidAlive: () => true });
    fs.writeFileSync(lockPath, JSON.stringify({ instanceId: 'new-owner', pid: 202, port: 3000 }), 'utf8');

    first.release();

    assert.strictEqual(fs.existsSync(lockPath), true);
});
