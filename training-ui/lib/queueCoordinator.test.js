const assert = require('assert');
const test = require('node:test');

const { createQueueCoordinator } = require('./queueCoordinator');

function makeTimers() {
    const timers = [];
    return {
        timers,
        setTimer(callback, delay) {
            const timer = { callback, delay, cancelled: false, unref() {} };
            timers.push(timer);
            return timer;
        },
        clearTimer(timer) {
            timer.cancelled = true;
        }
    };
}

test('stale running-process detection retries until the next job can start', async () => {
    const fakeTimers = makeTimers();
    const detectedJobs = ['previous-job', null];
    const starts = [];
    const coordinator = createQueueCoordinator({
        isEnabled: () => true,
        getRunningJob: async () => detectedJobs.shift(),
        getNextJob: () => 'next-job',
        startJob: async (jobName) => starts.push(jobName),
        retryDelayMs: 1500,
        setTimer: fakeTimers.setTimer,
        clearTimer: fakeTimers.clearTimer
    });

    const firstAttempt = await coordinator.advanceNow();

    assert.strictEqual(firstAttempt.status, 'waiting');
    assert.strictEqual(coordinator.hasScheduledAdvance(), true);
    assert.strictEqual(fakeTimers.timers[0].delay, 1500);
    assert.deepStrictEqual(starts, []);

    await fakeTimers.timers[0].callback();

    assert.deepStrictEqual(starts, ['next-job']);
    assert.strictEqual(coordinator.hasScheduledAdvance(), false);
});

test('repeated advance requests share one pending timer', () => {
    const fakeTimers = makeTimers();
    const coordinator = createQueueCoordinator({
        isEnabled: () => true,
        getRunningJob: async () => null,
        getNextJob: () => 'next-job',
        startJob: async () => {},
        setTimer: fakeTimers.setTimer,
        clearTimer: fakeTimers.clearTimer
    });

    assert.strictEqual(coordinator.requestAdvance(1000), true);
    assert.strictEqual(coordinator.requestAdvance(1000), false);
    assert.strictEqual(fakeTimers.timers.length, 1);
});

test('cancel prevents a scheduled transition from running', async () => {
    const fakeTimers = makeTimers();
    const starts = [];
    let enabled = true;
    const coordinator = createQueueCoordinator({
        isEnabled: () => enabled,
        getRunningJob: async () => null,
        getNextJob: () => 'next-job',
        startJob: async (jobName) => starts.push(jobName),
        setTimer: fakeTimers.setTimer,
        clearTimer: fakeTimers.clearTimer
    });

    coordinator.requestAdvance(1000);
    enabled = false;
    coordinator.cancel();

    assert.strictEqual(fakeTimers.timers[0].cancelled, true);
    assert.strictEqual(coordinator.hasScheduledAdvance(), false);
    assert.deepStrictEqual(starts, []);
});

test('empty queue reports completion exactly once', async () => {
    let emptyCalls = 0;
    const coordinator = createQueueCoordinator({
        isEnabled: () => true,
        getRunningJob: async () => null,
        getNextJob: () => null,
        startJob: async () => {},
        onQueueEmpty: () => { emptyCalls += 1; }
    });

    const result = await coordinator.advanceNow();

    assert.strictEqual(result.status, 'empty');
    assert.strictEqual(emptyCalls, 1);
});
