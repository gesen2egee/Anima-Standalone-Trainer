const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');

const { createJobQueue } = require('./jobQueue');

function makeQueue(existingJobs = ['alpha', 'beta', 'gamma']) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'anima-queue-'));
    const statePath = path.join(dir, 'queue_state.json');
    const queue = createJobQueue({
        statePath,
        jobExists: (name) => existingJobs.includes(name)
    });
    return { dir, statePath, queue };
}

test('completed active job leaves queue so it can return to pending jobs', () => {
    const { queue } = makeQueue();

    queue.enqueue('alpha');
    queue.enqueue('beta');
    queue.markActive('alpha');
    queue.completeActive('alpha');

    assert.deepStrictEqual(queue.getState().items, ['beta']);
    assert.strictEqual(queue.getState().active, null);
    assert.deepStrictEqual(queue.getPendingJobNames(['alpha', 'beta', 'gamma']), ['alpha', 'gamma']);
});

test('new queued jobs append to the bottom and duplicates are ignored', () => {
    const { queue } = makeQueue();

    queue.enqueue('alpha');
    queue.enqueue('beta');
    queue.enqueue('alpha');

    assert.deepStrictEqual(queue.getState().items, ['alpha', 'beta']);
});

test('queue state persists to disk', () => {
    const { statePath, queue } = makeQueue();

    queue.enqueue('alpha');
    queue.enqueue('beta');
    queue.markActive('alpha');

    const restored = createJobQueue({
        statePath,
        jobExists: (name) => ['alpha', 'beta'].includes(name)
    });

    assert.deepStrictEqual(restored.getState().items, ['alpha', 'beta']);
    assert.strictEqual(restored.getState().active, 'alpha');
});

test('clearing active keeps the job queued for a later retry', () => {
    const { queue } = makeQueue();

    queue.enqueue('alpha');
    queue.markActive('alpha');
    queue.clearActive('alpha');

    assert.deepStrictEqual(queue.getState().items, ['alpha']);
    assert.strictEqual(queue.getState().active, null);
});

test('successful queued job completion removes the job even when active marker is missing', () => {
    const { queue } = makeQueue();

    queue.enqueue('alpha');
    queue.enqueue('beta');
    queue.finishQueuedJob('alpha', { success: true });

    assert.deepStrictEqual(queue.getState().items, ['beta']);
    assert.strictEqual(queue.getState().active, null);
});

test('failed queued job completion clears active but keeps the job for retry', () => {
    const { queue } = makeQueue();

    queue.enqueue('alpha');
    queue.enqueue('beta');
    queue.markActive('alpha');
    queue.finishQueuedJob('alpha', { success: false });

    assert.deepStrictEqual(queue.getState().items, ['alpha', 'beta']);
    assert.strictEqual(queue.getState().active, null);
});
