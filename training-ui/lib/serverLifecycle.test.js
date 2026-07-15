const assert = require('assert');
const test = require('node:test');

const {
    isExpectedNodeServer,
    monitorParentProcess,
    stopPreviousNodeOnPort
} = require('./serverLifecycle');

test('only a Node server.js process is recognized as the previous UI', () => {
    const nodeInfo = () => ({ name: 'node.exe', commandLine: 'node server.js --port=3000' });
    const pythonInfo = () => ({ name: 'python.exe', commandLine: 'python server.py' });

    assert.strictEqual(isExpectedNodeServer(101, 'C:/app/server.js', nodeInfo), true);
    assert.strictEqual(isExpectedNodeServer(102, 'C:/app/server.js', pythonInfo), false);
});

test('port takeover terminates only a verified previous Node UI', () => {
    const terminated = [];
    const result = stopPreviousNodeOnPort({
        port: 3000,
        serverScriptPath: 'C:/app/server.js',
        findOwner: () => 101,
        isExpected: () => true,
        terminate: pid => { terminated.push(pid); return true; }
    });

    assert.deepStrictEqual(result, { stopped: true, pid: 101 });
    assert.deepStrictEqual(terminated, [101]);
});

test('parent monitor calls shutdown after Python disappears', async () => {
    let checks = 0;
    const missingPid = await new Promise(resolve => {
        monitorParentProcess(321, resolve, {
            intervalMs: 5,
            isAlive: () => checks++ === 0
        });
    });

    assert.strictEqual(missingPid, 321);
});
